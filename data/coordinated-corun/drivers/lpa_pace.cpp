// Paced prefill / decode loop for the coordinated-corun study (design D1).
//
// Uses only the public llama.h / ggml-cpu.h API of llama.cpp 8345f33 and mirrors llama-bench's
// test_prompt / test_gen (fixed pseudo-random token ids from an unseeded std::rand, BOS first,
// logits for the last token of a batch, one warm-up run, a ggml threadpool of -t threads with
// llama-bench's default poll 50). After every ubatch (prefill) or token (decode) it calls
// llama_synchronize, takes a timestamp, then sleeps t_step x (1/u - 1), t_step being the measured
// time of that step, so the loop computes a fraction u of the wall time.
//
//   lpa_pace -m <gguf> -ngl <n> -t <threads> --mode prefill|decode -P <prompt|depth> -C <ubatch>
//            -u <duty> -s <seconds> [-fa auto|0|1] [-dev none]
//
// prefill: clear the KV memory, decode P tokens in batches of C (n_batch = n_ubatch = C), repeat.
// decode:  fill depth P once (unmeasured, batches of C), then one token at a time, removing back
//          to depth P every 128 tokens. A new repetition (prefill) or token (decode) starts only
//          before the deadline, so a prefill repetition is always whole.
// stdout: one JSONL line per step {"rep","step","start","end","gap_end","tokens"} (epoch seconds),
// then a summary line {"summary":1, ..., "rate_compute","rate_wall","duty"}: rates over the summed
// step times and over the wall time from the first step's start to the end of the last step's gap,
// and duty = compute / wall. Progress goes to stderr.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "ggml-cpu.h"
#include "llama.h"

using clk = std::chrono::steady_clock;

static double g_epoch0;  // epoch seconds at g_t0
static clk::time_point g_t0;

static double now_epoch() {
  return g_epoch0 + std::chrono::duration<double>(clk::now() - g_t0).count();
}

static void quiet_log(ggml_log_level level, const char* text, void*) {
  if (level == GGML_LOG_LEVEL_ERROR) {
    fputs(text, stderr);
  }
}

static void usage() {
  fprintf(stderr,
          "usage: lpa_pace -m <gguf> -ngl <n> -t <threads> --mode prefill|decode -P <n> -C <n> "
          "-u <duty> -s <seconds> [-fa auto|0|1] [-dev none]\n");
  exit(2);
}

int main(int argc, char** argv) {
  std::string model_path, mode, fa = "auto";
  int ngl = -1, n_threads = 1, P = -1, C = 512;
  double u = 1.0, secs = 30;
  bool dev_none = false;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (i + 1 >= argc) usage();
    std::string v = argv[++i];
    if (a == "-m")
      model_path = v;
    else if (a == "-ngl")
      ngl = atoi(v.c_str());
    else if (a == "-t")
      n_threads = atoi(v.c_str());
    else if (a == "--mode")
      mode = v;
    else if (a == "-P")
      P = atoi(v.c_str());
    else if (a == "-C")
      C = atoi(v.c_str());
    else if (a == "-u")
      u = atof(v.c_str());
    else if (a == "-s")
      secs = atof(v.c_str());
    else if (a == "-fa")
      fa = v;
    else if (a == "-dev" && v == "none")
      dev_none = true;
    else
      usage();
  }
  if (model_path.empty() || ngl < 0 || P < 0 || C <= 0 || u <= 0 || u > 1 ||
      (mode != "prefill" && mode != "decode") || (mode == "prefill" && P == 0)) {
    usage();
  }
  const bool prefill = mode == "prefill";

  g_t0 = clk::now();
  g_epoch0 =
      std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();

  llama_log_set(quiet_log, nullptr);
  ggml_backend_load_all();
  llama_backend_init();

  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = ngl;
  ggml_backend_dev_t no_devices[1] = {nullptr};
  if (dev_none) mp.devices = no_devices;  // as llama-bench -dev none: CPU only
  llama_model* model = llama_model_load_from_file(model_path.c_str(), mp);
  if (!model) {
    fprintf(stderr, "error: failed to load %s\n", model_path.c_str());
    return 1;
  }

  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = prefill ? P : P + 128;  // llama-bench: n_prompt + n_gen + n_depth
  cp.n_batch = C;
  cp.n_ubatch = C;
  cp.flash_attn_type = fa == "auto" ? LLAMA_FLASH_ATTN_TYPE_AUTO
                       : fa == "1"  ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                    : LLAMA_FLASH_ATTN_TYPE_DISABLED;
  llama_context* ctx = llama_init_from_model(model, cp);
  if (!ctx) {
    fprintf(stderr, "error: failed to create context\n");
    return 1;
  }
  ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
  tpp.poll = 50;
  ggml_threadpool* tp = ggml_threadpool_new(&tpp);
  llama_attach_threadpool(ctx, tp, nullptr);
  llama_set_n_threads(ctx, n_threads, n_threads);

  const llama_vocab* vocab = llama_model_get_vocab(model);
  const int n_vocab = llama_vocab_n_tokens(vocab);
  const bool add_bos = llama_vocab_get_add_bos(vocab);
  llama_memory_t mem = llama_get_memory(ctx);
  llama_batch batch = llama_batch_init(C, 0, 1);

  // Decode n tokens at positions pos.. as one batch (logits for the last), then synchronize.
  auto run = [&](int pos, int n, bool first) {
    batch.n_tokens = n;
    for (int i = 0; i < n; i++) {
      batch.token[i] = i == 0 && first && add_bos ? llama_vocab_bos(vocab) : std::rand() % n_vocab;
      batch.pos[i] = pos + i;
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = 0;
      batch.logits[i] = i == n - 1;
    }
    if (llama_decode(ctx, batch) != 0) {
      fprintf(stderr, "error: llama_decode failed at pos %d\n", pos);
      exit(1);
    }
    llama_synchronize(ctx);
  };
  double compute = 0, first_start = -1, wall_end = 0;
  long tokens = 0, steps = 0;
  int reps = 0;
  const double gap_factor = 1.0 / u - 1.0;
  auto step = [&](int rep, int idx, int pos, int n, bool first) {
    const double s = now_epoch();
    run(pos, n, first);
    const double e = now_epoch();
    if (first_start < 0) first_start = s;
    compute += e - s;
    tokens += n;
    steps++;
    if (gap_factor > 0)
      std::this_thread::sleep_for(std::chrono::duration<double>((e - s) * gap_factor));
    wall_end = now_epoch();  // the wall time of a step includes its gap
    printf("{\"rep\":%d,\"step\":%d,\"start\":%.6f,\"end\":%.6f,\"gap_end\":%.6f,\"tokens\":%d}\n",
           rep,
           idx,
           s,
           e,
           wall_end,
           n);
  };

  // Warm-up (unmeasured): one whole prompt, or the depth fill plus one token.
  llama_memory_clear(mem, false);
  for (int done = 0; done < P; done += C) run(done, std::min(C, P - done), done == 0);
  if (!prefill) {
    run(P, 1, true);
    llama_memory_seq_rm(mem, 0, P, -1);
  }
  fprintf(stderr,
          "lpa_pace: warm-up done, measuring %s P=%d C=%d u=%.3f for %.0f s\n",
          mode.c_str(),
          P,
          C,
          u,
          secs);

  const double deadline = now_epoch() + secs;
  if (prefill) {
    while (now_epoch() < deadline) {
      llama_memory_clear(mem, false);
      for (int done = 0, i = 0; done < P; done += C, i++)
        step(reps, i, done, std::min(C, P - done), done == 0);
      reps++;
    }
  } else {
    int pos = P;
    while (now_epoch() < deadline) {
      if (pos == P + 128) {
        llama_memory_seq_rm(mem, 0, P, -1);
        pos = P;
        reps++;
      }
      step(reps, pos - P, pos, 1, pos == P);  // BOS first, as test_gen
      pos++;
    }
    reps++;
  }
  const double wall = wall_end - first_start;
  printf(
      "{\"summary\":1,\"mode\":\"%s\",\"P\":%d,\"C\":%d,\"u\":%.4f,\"ngl\":%d,\"threads\":%d,"
      "\"fa\":\"%s\","
      "\"dev_none\":%d,\"reps\":%d,\"steps\":%ld,\"tokens\":%ld,\"compute_s\":%.6f,\"wall_s\":%.6f,"
      "\"rate_compute\":%.4f,\"rate_wall\":%.4f,\"duty\":%.4f}\n",
      mode.c_str(),
      P,
      C,
      u,
      ngl,
      n_threads,
      fa.c_str(),
      dev_none,
      reps,
      steps,
      tokens,
      compute,
      wall,
      static_cast<double>(tokens) / compute,
      static_cast<double>(tokens) / wall,
      compute / wall);
  fflush(stdout);

  llama_batch_free(batch);
  llama_free(ctx);
  ggml_threadpool_free(tp);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
