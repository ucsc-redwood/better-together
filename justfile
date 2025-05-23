#  ----------------------------------------------------------------------------
#  Setup Configuration
#  ----------------------------------------------------------------------------

# Set configuration for Android devices (on a machine using ADB)
# drwxrwxr-x 11 doremy doremy 4.0K Oct 16 12:23 26.1.10909125/
# drwxrwxr-x 11 doremy doremy 4.0K Oct 16 12:41 27.0.12077973/
# drwxrwxr-x 11 doremy doremy 4.0K Oct 16 13:12 28.0.12433566/
# drwxrwxr-x 11 doremy doremy 4.0K Dec  7 12:08 28.0.12674087/
# drwxrwxr-x 11 doremy doremy 4.0K Feb 17 00:08 28.0.13004108/
# drwxrwxr-x 11 doremy doremy 4.0K Mar 10 12:24 29.0.13113456/
set-android:
    xmake f -p android -a arm64-v8a --ndk=~/Android/Sdk/ndk/29.0.13113456/ --android_sdk=~/Android/Sdk/ --ndk_sdkver=29 -c -v --use_vulkan=yes --use_cuda=no -m release

# Set configuration for NVIDIA Jetson Orin
set-jetson:
    xmake f -p linux -a arm64 --use_cuda=yes --use_vulkan=no -c -v -m release

# Set default configuration for PC
set-default:
    xmake f -p linux -a x86_64 -c -v --use_vulkan=no --use_cuda=yes -m release

#  ----------------------------------------------------------------------------
#  Compile Shaders
#  ----------------------------------------------------------------------------

# Compile Vulkan shader (need xxd)
compile-shader:
    make

compile_commands:
    xmake project -k compile_commands
    sed -i 's/"-rdc=true",//g' compile_commands.json

#  ----------------------------------------------------------------------------
#  Benchmark Related
#  ----------------------------------------------------------------------------

# Remove all temporary files from Android devices, then push resources folder to devices
rm-android-tmp:
    adb -s 3A021JEHN02756 shell "rm -rf /data/local/tmp/*"
    adb -s 9b034f1b shell "rm -rf /data/local/tmp/*"
    adb -s ce0717178d7758b00b7e shell "rm -rf /data/local/tmp/*"
    
    xmake push-all-resources

# List all files in the temporary directory of Android devices
cat-android-tmp:
    adb -s 3A021JEHN02756 shell "ls -la /data/local/tmp"
    adb -s 9b034f1b shell "ls -la /data/local/tmp"
    adb -s ce0717178d7758b00b7e shell "ls -la /data/local/tmp"

#  ----------------------------------------------------------------------------
#  Run benchmarks 
#  ----------------------------------------------------------------------------

run-jetson-cu-bm:
    xmake r bm-cifar-dense-cu --device jetson
    xmake r bm-cifar-sparse-cu --device jetson
    xmake r bm-tree-cu --device jetson

run-jetson-omp-bm:
    xmake r bm-cifar-dense-omp --device jetson
    xmake r bm-cifar-sparse-omp --device jetson
    xmake r bm-tree-omp --device jetson

run-jetson-bm:
    just run-jetson-cu-bm
    just run-jetson-omp-bm

run-jetsonlowpower-cu-bm:
    xmake r bm-cifar-dense-cu --device jetsonlowpower
    xmake r bm-cifar-sparse-cu --device jetsonlowpower
    xmake r bm-tree-cu --device jetsonlowpower

run-jetsonlowpower-omp-bm:
    xmake r bm-cifar-dense-omp --device jetsonlowpower
    xmake r bm-cifar-sparse-omp --device jetsonlowpower
    xmake r bm-tree-omp --device jetsonlowpower

run-jetsonlowpower-bm:
    just run-jetsonlowpower-cu-bm
    just run-jetsonlowpower-omp-bm

run-android-vk-bm:
    python3 scripts/collect_android_benchmarks.py --device 3A021JEHN02756 --benchmark bm-cifar-dense-vk
    python3 scripts/collect_android_benchmarks.py --device 3A021JEHN02756 --benchmark bm-cifar-sparse-vk
    python3 scripts/collect_android_benchmarks.py --device 3A021JEHN02756 --benchmark bm-tree-vk

    python3 scripts/collect_android_benchmarks.py --device 9b034f1b --benchmark bm-cifar-dense-vk
    python3 scripts/collect_android_benchmarks.py --device 9b034f1b --benchmark bm-cifar-sparse-vk
    python3 scripts/collect_android_benchmarks.py --device 9b034f1b --benchmark bm-tree-vk

run-android-omp-bm:
    python3 scripts/collect_android_benchmarks.py --device 3A021JEHN02756 --benchmark bm-cifar-dense-omp
    python3 scripts/collect_android_benchmarks.py --device 3A021JEHN02756 --benchmark bm-cifar-sparse-omp
    python3 scripts/collect_android_benchmarks.py --device 3A021JEHN02756 --benchmark bm-tree-omp

    python3 scripts/collect_android_benchmarks.py --device 9b034f1b --benchmark bm-cifar-dense-omp
    python3 scripts/collect_android_benchmarks.py --device 9b034f1b --benchmark bm-cifar-sparse-omp
    python3 scripts/collect_android_benchmarks.py --device 9b034f1b --benchmark bm-tree-omp

run-android-bm:
    just run-android-vk-bm
    just run-android-omp-bm

#  ----------------------------------------------------------------------------
#  from google benchmark output (json) to schedules (json)
#  ----------------------------------------------------------------------------

db-to-schedules:
    python3 scripts/gen_schedules.py -d jetson -a CifarDense -b ./data/stable_bm_out_v3/ -o ./data/schedule_files_v3 --top 50
    python3 scripts/gen_schedules.py -d jetson -a CifarSparse -b ./data/stable_bm_out_v3/ -o ./data/schedule_files_v3 --top 50
    python3 scripts/gen_schedules.py -d jetson -a Tree -b ./data/stable_bm_out_v3/ -o ./data/schedule_files_v3 --top 50
    # python3 scripts/gen_schedules.py -d jetsonlowpower -a CifarDense -b ./data/stable_bm_out_v3/ -o ./data/schedule_files_v3 --top 50
    # python3 scripts/gen_schedules.py -d jetsonlowpower -a CifarSparse -b ./data/stable_bm_out_v3/ -o ./data/schedule_files_v3 --top 50
    # python3 scripts/gen_schedules.py -d jetsonlowpower -a Tree -b ./data/stable_bm_out_v3/ -o ./data/schedule_files_v3 --top 50
    # python3 scripts/gen_schedules.py -d 3A021JEHN02756 -a CifarDense -b ./data/stable_bm_out_v3/ -o ./data/schedule_files_v3 --top 50
    # python3 scripts/gen_schedules.py -d 3A021JEHN02756 -a CifarSparse -b ./data/stable_bm_out_v3/ -o ./data/schedule_files_v3 --top 50
    # python3 scripts/gen_schedules.py -d 3A021JEHN02756 -a Tree -b ./data/stable_bm_out_v3/ -o ./data/schedule_files_v3 --top 50
    # python3 scripts/gen_schedules.py -d 9b034f1b -a CifarDense -b ./data/stable_bm_out_v3/ -o ./data/schedule_files_v3 --top 50
    # python3 scripts/gen_schedules.py -d 9b034f1b -a CifarSparse -b ./data/stable_bm_out_v3/ -o ./data/schedule_files_v3 --top 50
    # python3 scripts/gen_schedules.py -d 9b034f1b -a Tree -b ./data/stable_bm_out_v3/ -o ./data/schedule_files_v3 --top 50

schedules-to-code-new:
    # python3 scripts/codegen/new_vk.py data/schedule_files_v3/ CifarDense pipe/new-cifar-dense-vk/generated_code.hpp
    # python3 scripts/codegen/new_vk.py data/schedule_files_v3/ CifarSparse pipe/new-cifar-sparse-vk/generated_code.hpp
    # python3 scripts/codegen/new_vk.py data/schedule_files_v3/ Tree pipe/new-tree-vk/generated_code.hpp

    python3 scripts/codegen/new_cu.py data/schedule_files_v3/ CifarDense pipe/new-cifar-dense-cu/generated_code.cuh
    python3 scripts/codegen/new_cu.py data/schedule_files_v3/ CifarSparse pipe/new-cifar-sparse-cu/generated_code.cuh
    python3 scripts/codegen/new_cu.py data/schedule_files_v3/ Tree pipe/new-tree-cu/generated_code.cuh

    python3 scripts/codegen/new_cu_non_bm.py data/schedule_files_v3/ CifarDense pipe/new-cifar-dense-cu/generated_code_non_bm.cuh
    python3 scripts/codegen/new_cu_non_bm.py data/schedule_files_v3/ CifarSparse pipe/new-cifar-sparse-cu/generated_code_non_bm.cuh
    python3 scripts/codegen/new_cu_non_bm.py data/schedule_files_v3/ Tree pipe/new-tree-cu/generated_code_non_bm.cuh

    xmake format

analysis:
     python3 scripts/analysis/parse_benchmark.py --schedule-root data/schedule_files_v3/ data/pipe_out/3A021JEHN02756_CifarDense.txt --sort-by avg_time
     python3 scripts/analysis/parse_benchmark.py --schedule-root data/schedule_files_v3/ data/pipe_out/3A021JEHN02756_CifarSparse.txt --sort-by avg_time
     python3 scripts/analysis/parse_benchmark.py --schedule-root data/schedule_files_v3/ data/pipe_out/3A021JEHN02756_Tree.txt --sort-by avg_time
     python3 scripts/analysis/parse_benchmark.py --schedule-root data/schedule_files_v3/ data/pipe_out/9b034f1b_CifarDense.txt --sort-by avg_time
     python3 scripts/analysis/parse_benchmark.py --schedule-root data/schedule_files_v3/ data/pipe_out/9b034f1b_CifarSparse.txt --sort-by avg_time
     python3 scripts/analysis/parse_benchmark.py --schedule-root data/schedule_files_v3/ data/pipe_out/9b034f1b_Tree.txt --sort-by avg_time
     python3 scripts/analysis/parse_benchmark.py --schedule-root data/schedule_files_v3/ data/pipe_out/jetson_CifarDense.txt --sort-by avg_time
     python3 scripts/analysis/parse_benchmark.py --schedule-root data/schedule_files_v3/ data/pipe_out/jetson_CifarSparse.txt --sort-by avg_time
     python3 scripts/analysis/parse_benchmark.py --schedule-root data/schedule_files_v3/ data/pipe_out/jetson_Tree.txt --sort-by avg_time
     python3 scripts/analysis/parse_benchmark.py --schedule-root data/schedule_files_v3/ data/pipe_out/jetsonlowpower_CifarDense.txt --sort-by avg_time
     python3 scripts/analysis/parse_benchmark.py --schedule-root data/schedule_files_v3/ data/pipe_out/jetsonlowpower_CifarSparse.txt --sort-by avg_time
     python3 scripts/analysis/parse_benchmark.py --schedule-root data/schedule_files_v3/ data/pipe_out/jetsonlowpower_Tree.txt --sort-by avg_time





# ----------------------------------------------------------------------------
# Tmp
# ----------------------------------------------------------------------------


# This is ued to generate pipeline graph
run-benchmarks:
    xmake r gen-record-pipe-cifar-sparse-vk --schedule 1 | tee BM_pipe_cifar_sparse_vk_schedule_1.raw.txt
    xmake r gen-record-pipe-cifar-sparse-vk --schedule 2 | tee BM_pipe_cifar_sparse_vk_schedule_2.raw.txt
    xmake r gen-record-pipe-cifar-sparse-vk --schedule 3 | tee BM_pipe_cifar_sparse_vk_schedule_3.raw.txt
    xmake r gen-record-pipe-cifar-sparse-vk --schedule 4 | tee BM_pipe_cifar_sparse_vk_schedule_4.raw.txt
    xmake r gen-record-pipe-cifar-sparse-vk --schedule 5 | tee BM_pipe_cifar_sparse_vk_schedule_5.raw.txt
    xmake r gen-record-pipe-cifar-sparse-vk --schedule 6 | tee BM_pipe_cifar_sparse_vk_schedule_6.raw.txt
    xmake r gen-record-pipe-cifar-sparse-vk --schedule 7 | tee BM_pipe_cifar_sparse_vk_schedule_7.raw.txt
    xmake r gen-record-pipe-cifar-sparse-vk --schedule 8 | tee BM_pipe_cifar_sparse_vk_schedule_8.raw.txt
    xmake r gen-record-pipe-cifar-sparse-vk --schedule 9 | tee BM_pipe_cifar_sparse_vk_schedule_9.raw.txt
    xmake r gen-record-pipe-cifar-sparse-vk --schedule 10 | tee BM_pipe_cifar_sparse_vk_schedule_10.raw.txt

# This is used to compute the real measurement of the pipeline schedule
run-best-benchmarks:
    rm BM_best_raw.txt
    touch BM_best_raw.txt
    xmake r bm-best-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_1/ | tee -a BM_best_raw.txt
    xmake r bm-best-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_2/ | tee -a BM_best_raw.txt
    xmake r bm-best-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_3/ | tee -a BM_best_raw.txt
    xmake r bm-best-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_4/ | tee -a BM_best_raw.txt
    xmake r bm-best-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_5/ | tee -a BM_best_raw.txt
    xmake r bm-best-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_6/ | tee -a BM_best_raw.txt
    xmake r bm-best-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_7/ | tee -a BM_best_raw.txt
    xmake r bm-best-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_8/ | tee -a BM_best_raw.txt
    xmake r bm-best-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_9/ | tee -a BM_best_raw.txt
    xmake r bm-best-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_10/ | tee -a BM_best_raw.txt

# # This is used to generate benchmark Table, (used to generate math model)
# run-best-benchmarks-table:
#     rm -f BM_benchmark_table.txt
#     touch BM_benchmark_table.txt
#     xmake r bm-pipe-cifar-sparse-vk -l off --benchmark_filter=VK/CifarSparse/Baseline/iterations:1/ | tee -a BM_benchmark_table.txt
#     xmake r bm-pipe-cifar-sparse-vk -l off --benchmark_filter=VK/CifarSparse/Baseline/iterations:2/ | tee -a BM_benchmark_table.txt
#     xmake r bm-pipe-cifar-sparse-vk -l off --benchmark_filter=VK/CifarSparse/Baseline/iterations:3/ | tee -a BM_benchmark_table.txt
#     xmake r bm-pipe-cifar-sparse-vk -l off --benchmark_filter=VK/CifarSparse/Baseline/iterations:4/ | tee -a BM_benchmark_table.txt
#     xmake r bm-pipe-cifar-sparse-vk -l off --benchmark_filter=VK/CifarSparse/Baseline/iterations:5/ | tee -a BM_benchmark_table.txt
#     xmake r bm-pipe-cifar-sparse-vk -l off --benchmark_filter=VK/CifarSparse/Baseline/iterations:6/ | tee -a BM_benchmark_table.txt
#     xmake r bm-pipe-cifar-sparse-vk -l off --benchmark_filter=VK/CifarSparse/Baseline/iterations:7/ | tee -a BM_benchmark_table.txt
#     xmake r bm-pipe-cifar-sparse-vk -l off --benchmark_filter=VK/CifarSparse/Baseline/iterations:8/ | tee -a BM_benchmark_table.txt
#     xmake r bm-pipe-cifar-sparse-vk -l off --benchmark_filter=VK/CifarSparse/Baseline/iterations:9/ | tee -a BM_benchmark_table.txt
#     xmake r bm-pipe-cifar-sparse-vk -l off --benchmark_filter=VK/CifarSparse/Baseline/iterations:10/ | tee -a BM_benchmark_table.txt

# -----------------------------------------------------------------------------
# Target 2: Process the raw output using sed
# This extracts lines 8 to 1707 and removes the unwanted "Chunk 3" blocks with zero times
# -----------------------------------------------------------------------------
process-results:
	@echo "Processing raw benchmark outputs..."
	sed -n '11,1710p' BM_pipe_cifar_sparse_vk_schedule_1.raw.txt | sed '/^  Chunk 3:$/{N;N;N;/^  Chunk 3:\n    Start: 0 us\n    End: 0 us\n    Duration: 0 us$/d}' > BM_pipe_cifar_sparse_vk_schedule_1.txt
	sed -n '11,1710p' BM_pipe_cifar_sparse_vk_schedule_2.raw.txt | sed '/^  Chunk 3:$/{N;N;N;/^  Chunk 3:\n    Start: 0 us\n    End: 0 us\n    Duration: 0 us$/d}' > BM_pipe_cifar_sparse_vk_schedule_2.txt
	sed -n '11,1710p' BM_pipe_cifar_sparse_vk_schedule_3.raw.txt | sed '/^  Chunk 3:$/{N;N;N;/^  Chunk 3:\n    Start: 0 us\n    End: 0 us\n    Duration: 0 us$/d}' > BM_pipe_cifar_sparse_vk_schedule_3.txt
	sed -n '11,1710p' BM_pipe_cifar_sparse_vk_schedule_4.raw.txt | sed '/^  Chunk 3:$/{N;N;N;/^  Chunk 3:\n    Start: 0 us\n    End: 0 us\n    Duration: 0 us$/d}' > BM_pipe_cifar_sparse_vk_schedule_4.txt
	sed -n '11,1710p' BM_pipe_cifar_sparse_vk_schedule_5.raw.txt | sed '/^  Chunk 3:$/{N;N;N;/^  Chunk 3:\n    Start: 0 us\n    End: 0 us\n    Duration: 0 us$/d}' > BM_pipe_cifar_sparse_vk_schedule_5.txt
	sed -n '11,1710p' BM_pipe_cifar_sparse_vk_schedule_6.raw.txt | sed '/^  Chunk 3:$/{N;N;N;/^  Chunk 3:\n    Start: 0 us\n    End: 0 us\n    Duration: 0 us$/d}' > BM_pipe_cifar_sparse_vk_schedule_6.txt
	sed -n '11,1710p' BM_pipe_cifar_sparse_vk_schedule_7.raw.txt | sed '/^  Chunk 3:$/{N;N;N;/^  Chunk 3:\n    Start: 0 us\n    End: 0 us\n    Duration: 0 us$/d}' > BM_pipe_cifar_sparse_vk_schedule_7.txt
	sed -n '11,1710p' BM_pipe_cifar_sparse_vk_schedule_8.raw.txt | sed '/^  Chunk 3:$/{N;N;N;/^  Chunk 3:\n    Start: 0 us\n    End: 0 us\n    Duration: 0 us$/d}' > BM_pipe_cifar_sparse_vk_schedule_8.txt
	sed -n '11,1710p' BM_pipe_cifar_sparse_vk_schedule_9.raw.txt | sed '/^  Chunk 3:$/{N;N;N;/^  Chunk 3:\n    Start: 0 us\n    End: 0 us\n    Duration: 0 us$/d}' > BM_pipe_cifar_sparse_vk_schedule_9.txt
	sed -n '11,1710p' BM_pipe_cifar_sparse_vk_schedule_10.raw.txt | sed '/^  Chunk 3:$/{N;N;N;/^  Chunk 3:\n    Start: 0 us\n    End: 0 us\n    Duration: 0 us$/d}' > BM_pipe_cifar_sparse_vk_schedule_10.txt


# -----------------------------------------------------------------------------
# Target 3: Generate figures from processed data using Python.
# -----------------------------------------------------------------------------
make-fig:
	@echo "Generating figures..."
	python3 scripts-v2/analysis/pipe.py \
		scripts-v2/analysis/BM_pipe_cifar_sparse_vk_schedule_1.txt \
		-o task_execution_timeline_wide_cifar_sparse_vk_schedule_1 \
		--task-start 25 --task-end 50
	python3 scripts-v2/analysis/pipe.py \
		scripts-v2/analysis/BM_pipe_cifar_sparse_vk_schedule_2.txt \
		-o task_execution_timeline_wide_cifar_sparse_vk_schedule_2 \
		--task-start 25 --task-end 50
	python3 scripts-v2/analysis/pipe.py \
		scripts-v2/analysis/BM_pipe_cifar_sparse_vk_schedule_3.txt \
		-o task_execution_timeline_wide_cifar_sparse_vk_schedule_3 \
		--task-start 25 --task-end 50
	python3 scripts-v2/analysis/pipe.py \
		scripts-v2/analysis/BM_pipe_cifar_sparse_vk_schedule_4.txt \
		-o task_execution_timeline_wide_cifar_sparse_vk_schedule_4 \
		--task-start 25 --task-end 50
	python3 scripts-v2/analysis/pipe.py \
		scripts-v2/analysis/BM_pipe_cifar_sparse_vk_schedule_5.txt \
		-o task_execution_timeline_wide_cifar_sparse_vk_schedule_5 \
		--task-start 25 --task-end 50
	python3 scripts-v2/analysis/pipe.py \
		scripts-v2/analysis/BM_pipe_cifar_sparse_vk_schedule_6.txt \
		-o task_execution_timeline_wide_cifar_sparse_vk_schedule_6 \
		--task-start 25 --task-end 50
	python3 scripts-v2/analysis/pipe.py \
		scripts-v2/analysis/BM_pipe_cifar_sparse_vk_schedule_7.txt \
		-o task_execution_timeline_wide_cifar_sparse_vk_schedule_7 \
		--task-start 25 --task-end 50
	python3 scripts-v2/analysis/pipe.py \
		scripts-v2/analysis/BM_pipe_cifar_sparse_vk_schedule_8.txt \
		-o task_execution_timeline_wide_cifar_sparse_vk_schedule_8 \
		--task-start 25 --task-end 50
	python3 scripts-v2/analysis/pipe.py \
		scripts-v2/analysis/BM_pipe_cifar_sparse_vk_schedule_9.txt \
		-o task_execution_timeline_wide_cifar_sparse_vk_schedule_9 \
		--task-start 25 --task-end 50
	python3 scripts-v2/analysis/pipe.py \
		scripts-v2/analysis/BM_pipe_cifar_sparse_vk_schedule_10.txt \
		-o task_execution_timeline_wide_cifar_sparse_vk_schedule_10 \
		--task-start 25 --task-end 50

# -----------------------------------------------------------------------------
# "all" target to run every step in sequence.
# -----------------------------------------------------------------------------
all: 
    just run-benchmarks 
    just process-results 
    just make-fig


compare-full-and-non-full stage:
    # Run non-full benchmark
    xmake r bm-real-cifar-sparse-vk --stage {{stage}} -l off | tee non_full_stage_{{stage}}.txt

    # Run full benchmark
    xmake r bm-real-cifar-sparse-vk --stage {{stage}} -l off --full | tee full_stage_{{stage}}.txt

    # Extract and compare AVG metrics
    @echo "\n====== COMPARISON OF AVG METRICS ======"
    @echo "Processor | Non-Full (ms) | Full (ms)"
    @echo "--------------------------------------"
    @awk -F'|' '/PROCESSOR=/{split($1,p,"="); split($4,a,"="); printf "%s: %s ms\n", p[2], a[2]}' non_full_stage_{{stage}}.txt | sort
    @echo "--------------------------------------"
    @awk -F'|' '/PROCESSOR=/{split($1,p,"="); split($4,a,"="); printf "%s: %s ms\n", p[2], a[2]}' full_stage_{{stage}}.txt | sort

compare-all:
    just compare-full-and-non-full 1
    just compare-full-and-non-full 2
    just compare-full-and-non-full 3
    just compare-full-and-non-full 4
    just compare-full-and-non-full 5
    just compare-full-and-non-full 6
    just compare-full-and-non-full 7
    just compare-full-and-non-full 8
    just compare-full-and-non-full 9



# ----------------------------------------------------------------------------
# Measure Cifar-Sparse Real Time
# ----------------------------------------------------------------------------

try:
    xmake r bm-schedule-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_0/ | tee BM_pipe_cifar_sparse_vk_schedule_0.txt
    xmake r bm-schedule-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_1/ | tee BM_pipe_cifar_sparse_vk_schedule_1.txt
    xmake r bm-schedule-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_2/ | tee BM_pipe_cifar_sparse_vk_schedule_2.txt
    xmake r bm-schedule-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_3/ | tee BM_pipe_cifar_sparse_vk_schedule_3.txt
    xmake r bm-schedule-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_4/ | tee BM_pipe_cifar_sparse_vk_schedule_4.txt
    xmake r bm-schedule-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_5/ | tee BM_pipe_cifar_sparse_vk_schedule_5.txt
    xmake r bm-schedule-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_6/ | tee BM_pipe_cifar_sparse_vk_schedule_6.txt
    xmake r bm-schedule-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_7/ | tee BM_pipe_cifar_sparse_vk_schedule_7.txt
    xmake r bm-schedule-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_8/ | tee BM_pipe_cifar_sparse_vk_schedule_8.txt
    xmake r bm-schedule-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_9/ | tee BM_pipe_cifar_sparse_vk_schedule_9.txt
    xmake r bm-schedule-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_10/ | tee BM_pipe_cifar_sparse_vk_schedule_10.txt


# ----------------------------------------------------------------------------
# Measure Cifar-Sparse Real Time and Gen Figure
# ----------------------------------------------------------------------------

# py scripts-v2/gen_schedule/schedule.py

make_bm_log stage:
    xmake r gen-records-cifar-sparse-vk --schedule {{stage}} | sed -n '11,1710p' | tee BM_best_raw_stage_{{stage}}.txt

log_to_figure stage:
    python3 scripts-v2/analysis/gen_chunk_figure.py BM_best_raw_stage_{{stage}}.txt --output BM_best_raw_stage_{{stage}}.png --start-time 0 --end-time 1

try-all:
    just make_bm_log 1
    just make_bm_log 2
    just make_bm_log 3
    just make_bm_log 4
    just make_bm_log 5
    just make_bm_log 6
    just make_bm_log 7
    just make_bm_log 8
    just make_bm_log 9
    just make_bm_log 10


# Average task durations for each schedule:
# Schedule 0: 12.2012 ms
# Schedule 1: 12.416778 ms
# Schedule 2: 11.824737 ms
# Schedule 3: 7.610159 ms
# Schedule 4: 9.040559 ms
# Schedule 5: 12.960554 ms
# Schedule 6: 7.108262 ms
# Schedule 7: 12.148757 ms
# Schedule 8: 6.685347 ms
# Schedule 9: 7.954996 ms
# Schedule 10: 12.057925 ms
try-all-figure:
    just log_to_figure 1 | tee BM_best_raw_schedule_1_analysis.txt
    just log_to_figure 2 | tee BM_best_raw_schedule_2_analysis.txt
    just log_to_figure 3 | tee BM_best_raw_schedule_3_analysis.txt
    just log_to_figure 4 | tee BM_best_raw_schedule_4_analysis.txt
    just log_to_figure 5 | tee BM_best_raw_schedule_5_analysis.txt
    just log_to_figure 6 | tee BM_best_raw_schedule_6_analysis.txt
    just log_to_figure 7 | tee BM_best_raw_schedule_7_analysis.txt
    just log_to_figure 8 | tee BM_best_raw_schedule_8_analysis.txt
    just log_to_figure 9 | tee BM_best_raw_schedule_9_analysis.txt
    just log_to_figure 10 | tee BM_best_raw_schedule_10_analysis.txt
        cat BM_best_raw_schedule_*_analysis.txt | grep "All Tasks Average:" | awk -F'[()]' '{print $2}' | awk '{
                                                                        if (NR%4==1) max=$1
                                                                        if ($1>max) max=$1
                                                                        if (NR%4==0) print max
                                                                        }'
