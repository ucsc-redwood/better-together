#  ----------------------------------------------------------------------------
#  Setup Configuration
#  ----------------------------------------------------------------------------
# Requirements:
#   - xmake (modern build system)
#   - uv (modern python package manager)
#   - ssh
#   - adb (for Android)
#   - glslc (for compiling shaders)
#   - xxd (for generating header files from shaders)

# Set default configuration for PC
set-default:
    xmake f -p linux -a x86_64 -c -v --use_vulkan=no --use_cuda=yes -m release

# Set configuration for NVIDIA Jetson Orin Dev Kit
set-jetson:
    xmake f -p linux -a arm64 --use_cuda=yes --use_vulkan=yes -c -v -m release

# Set configuration for Android devices (on a machine using ADB)
# Download NDK from: https://developer.android.com/ndk/downloads
# 
#   wget https://dl.google.com/android/repository/android-ndk-r29-beta3-linux.zip
#   unzip android-ndk-r29-beta3-linux.zip
#   mv android-ndk-r29-beta3-linux ~/Android/
# 
# List of devices attached
#   3A021JEHN02756         device usb:9-1 product:lynx model:Pixel_7a device:lynx transport_id:4
#   9b034f1b               device usb:9-3 product:CPH2451 model:CPH2451 device:OP594DL1 transport_id:2
#   R5CY21Y3VEV            device usb:9-2 product:e2sxxx model:SM_S926B device:e2s transport_id:3
#   ZY22FLDDK7             device usb:9-4 product:ellis_retail model:moto_g_pure device:ellis transport_id:1
# 
# Tested and working versions:
#   26.1.10909125
#   27.0.12077973
#   28.0.12433566
#   28.0.12674087
#   28.0.13004108
#   29.0.13113456
#   29.0.13846066
#
# Set the NDK version in the justfile
#
set-android:
    xmake f -p android -a arm64-v8a --ndk=~/Android/android-ndk-r29-beta3 --ndk_sdkver=29 -c -v --use_vulkan=yes --use_cuda=no -m release

# Serving the generated schedules locally, so my android phones can access it via its IP address
# at port 8080
serve:
    uv run -m http.server --bind 0.0.0.0 --directory data/schedules/ 8080

# Used by client (e.g., my PC, connect to my ADB Android Server)
connect:
    ssh -N -f -L 5037:localhost:5037 yanwen@android-dev.ucsc

#  ----------------------------------------------------------------------------
#  Compile Shaders
#  ----------------------------------------------------------------------------

# Compile Vulkan shader (need xxd)
compile-shader:
    make

#  ----------------------------------------------------------------------------
#  Benchmark Related
#  ----------------------------------------------------------------------------

# Remove all temporary files from Android devices, then push resources folder to devices
rm-android-tmp:
    adb -s 3A021JEHN02756 shell "rm -rf /data/local/tmp/*"
    adb -s 9b034f1b shell "rm -rf /data/local/tmp/*"
    # adb -s ce0717178d7758b00b7e shell "rm -rf /data/local/tmp/*"
    adb -s R5CY21Y3VEV shell "rm -rf /data/local/tmp/*"
    
# List all files in the temporary directory of Android devices
cat-android-tmp:
    adb -s 3A021JEHN02756 shell "ls -la /data/local/tmp"
    adb -s 9b034f1b shell "ls -la /data/local/tmp"
    # adb -s ce0717178d7758b00b7e shell "ls -la /data/local/tmp"
    adb -s R5CY21Y3VEV shell "ls -la /data/local/tmp"


# ----------------------------------------------------------------------------
# Final Version
# ----------------------------------------------------------------------------

# Also we want to check affinity by 
#   xmake r test-affinity
#   xmake r test-affinity --device R5CY21Y3VEV
# 
#   xmake r bm-check-core-types 
#   xmake r bm-check-core-types --device R5CY21Y3VEV

# [1/3] Processing device: 3A021JEHN02756
# ---------------------------------------------------------------
# Benchmark                     Time             CPU   Iterations
# ---------------------------------------------------------------
# HeavyFloat/CoreID0/0        101 ms         99.2 ms            7
# HeavyFloat/CoreID1/1        100 ms         99.2 ms            7
# HeavyFloat/CoreID2/2        100 ms         99.1 ms            7
# HeavyFloat/CoreID3/3        100 ms         99.1 ms            7
# HeavyFloat/CoreID4/4       23.6 ms         23.5 ms           30
# HeavyFloat/CoreID5/5       23.6 ms         23.5 ms           30
# HeavyFloat/CoreID6/6       16.6 ms         16.5 ms           43
# HeavyFloat/CoreID7/7       17.2 ms         17.1 ms           43
# GraphBFS/CoreID0/0         6.67 ms         6.50 ms          115
# GraphBFS/CoreID1/1         6.47 ms         6.38 ms          112
# GraphBFS/CoreID2/2         6.29 ms         6.21 ms          120
# GraphBFS/CoreID3/3         6.37 ms         6.29 ms          114
# GraphBFS/CoreID4/4         2.03 ms         2.01 ms          346
# GraphBFS/CoreID5/5         2.04 ms         2.03 ms          346
# GraphBFS/CoreID6/6         1.07 ms         1.07 ms          666
# GraphBFS/CoreID7/7         1.08 ms         1.08 ms          644

# [2/3] Processing device: 9b034f1b
# ---------------------------------------------------------------
# Benchmark                     Time             CPU   Iterations
# ---------------------------------------------------------------
# HeavyFloat/CoreID0/0       88.0 ms         85.9 ms            8
# HeavyFloat/CoreID1/1       87.6 ms         85.8 ms            8
# HeavyFloat/CoreID2/2       87.0 ms         85.7 ms            8
# HeavyFloat/CoreID3/3       29.8 ms         29.6 ms           23
# HeavyFloat/CoreID4/4       29.8 ms         29.6 ms           24
# HeavyFloat/CoreID5/5       31.4 ms         31.2 ms           22
# HeavyFloat/CoreID6/6 ERROR OCCURRED: 'Failed to pin to core 6: Failed to pin thread to cores on Linux: 6'
# HeavyFloat/CoreID7/7       17.4 ms         17.4 ms           40
# GraphBFS/CoreID0/0         3.63 ms         3.56 ms          197
# GraphBFS/CoreID1/1         3.61 ms         3.55 ms          198
# GraphBFS/CoreID2/2         3.62 ms         3.56 ms          197
# GraphBFS/CoreID3/3         1.60 ms         1.59 ms          408
# GraphBFS/CoreID4/4         1.59 ms         1.58 ms          445
# GraphBFS/CoreID5/5         1.74 ms         1.72 ms          405
# GraphBFS/CoreID6/6   ERROR OCCURRED: 'Failed to pin to core 6: Failed to pin thread to cores on Linux: 6'
# GraphBFS/CoreID7/7         1.05 ms         1.04 ms          641

# [3/3] Processing device: R5CY21Y3VEV
# ---------------------------------------------------------------
# Benchmark                     Time             CPU   Iterations
# ---------------------------------------------------------------
# HeavyFloat/CoreID0/0        102 ms         99.9 ms            7
# HeavyFloat/CoreID1/1        101 ms         99.9 ms            7
# HeavyFloat/CoreID2/2        101 ms         99.9 ms            7
# HeavyFloat/CoreID3/3        101 ms         99.9 ms            7
# HeavyFloat/CoreID4/4       22.9 ms         22.8 ms           31
# HeavyFloat/CoreID5/5       22.9 ms         22.8 ms           31
# HeavyFloat/CoreID6/6       22.9 ms         22.8 ms           31
# HeavyFloat/CoreID7/7       20.3 ms         20.2 ms           35
# HeavyFloat/CoreID8/8       20.3 ms         20.2 ms           35
# HeavyFloat/CoreID9/9       13.8 ms         13.7 ms           51
# GraphBFS/CoreID0/0         4.98 ms         4.89 ms          138
# GraphBFS/CoreID1/1         5.08 ms         4.95 ms          145
# GraphBFS/CoreID2/2         4.81 ms         4.70 ms          148
# GraphBFS/CoreID3/3         4.89 ms         4.79 ms          147
# GraphBFS/CoreID4/4         1.39 ms         1.38 ms          526
# GraphBFS/CoreID5/5         1.37 ms         1.36 ms          532
# GraphBFS/CoreID6/6         1.39 ms         1.38 ms          523
# GraphBFS/CoreID7/7         1.27 ms         1.26 ms          577
# GraphBFS/CoreID8/8         1.26 ms         1.25 ms          581
# GraphBFS/CoreID9/9        0.812 ms        0.808 ms          861


# ----------------------------------------------------------------------------
# Run Baselines, it will show the CPU/GPU baseline performance in the terminal
# ----------------------------------------------------------------------------

# Run baselines on all Android devices
run-baselines-android:
    xmake r bm-baseline-cifar-dense-vk --device 3A021JEHN02756
    xmake r bm-baseline-cifar-sparse-vk --device 3A021JEHN02756
    xmake r bm-baseline-tree-vk --device 3A021JEHN02756
    xmake r bm-baseline-cifar-dense-vk --device 9b034f1b
    xmake r bm-baseline-cifar-sparse-vk --device 9b034f1b
    xmake r bm-baseline-tree-vk --device 9b034f1b

run-baselines-android-new:
    xmake r bm-baseline-cifar-dense-vk --device R5CY21Y3VEV
    xmake r bm-baseline-cifar-sparse-vk --device R5CY21Y3VEV
    xmake r bm-baseline-tree-vk --device R5CY21Y3VEV
    
run-baselines-jetson:
    xmake r bm-baseline-cifar-sparse-cu --device jetson
    xmake r bm-baseline-cifar-dense-cu --device jetson
    xmake r bm-baseline-tree-cu --device jetson
    xmake r bm-baseline-cifar-sparse-vk --device jetson
    xmake r bm-baseline-cifar-dense-vk --device jetson
    xmake r bm-baseline-tree-vk --device jetson

run-baselines-jetsonlowpower:
    xmake r bm-baseline-cifar-sparse-cu --device jetsonlowpower
    xmake r bm-baseline-cifar-dense-cu --device jetsonlowpower
    xmake r bm-baseline-tree-cu --device jetsonlowpower
    xmake r bm-baseline-cifar-sparse-vk --device jetsonlowpower
    xmake r bm-baseline-cifar-dense-vk --device jetsonlowpower
    xmake r bm-baseline-tree-vk --device jetsonlowpower

# ----------------------------------------------------------------------------
# BT-Profiler (Step 1)
# This will run the benchmark and collect the profiling data in the folder
# under 'data/bm_logs/<device>/<app>/<backend>/'
# e.g., data/bm_logs/3A021JEHN02756/tree/vk/
# 
# ----------------------------------------------------------------------------

# (Step 1) Collect all the data
collect device app backend:
    uv run scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app {{app}} --backend {{backend}} --device {{device}}

collect-android:
    just collect 3A021JEHN02756 tree vk
    just collect 3A021JEHN02756 cifar-sparse vk
    just collect 3A021JEHN02756 cifar-dense vk
    just collect 9b034f1b tree vk
    just collect 9b034f1b cifar-sparse vk
    just collect 9b034f1b cifar-dense vk

collect-android-new:
    just collect R5CY21Y3VEV tree vk
    just collect R5CY21Y3VEV cifar-sparse vk
    just collect R5CY21Y3VEV cifar-dense vk

collect-all-jetson:
    just collect jetson tree cu
    just collect jetson cifar-sparse cu
    just collect jetson cifar-dense cu

collect-android-all:
    just collect-android
    just collect-android
    just collect-android

# ----------------------------------------------------------------------------
# BT-Optimizer (Step 2)
# This will run the SMT solver to generate schedules, given the profiling data
# Output schedules are saved under the folder
# 'data/schedules/<device>/<app>/<backend>/'
# e.g., data/schedules/3A021JEHN02756/tree/vk/
# 
# The schedule file name is 'schedules_<table_type>_<minimize_mode>.json'
# e.g., schedules_btpm_gapness.json
# e.g., schedules_btpm_tmax.json
# 
# ----------------------------------------------------------------------------

# Generate schedules for all devices
gen-schedule device app backend table_type minimize_mode:
    uv run scripts/collect/02_gen_schedule_merged.py \
        --csv_root_folder data/bm_logs/ \
        --device {{device}} \
        --app {{app}} \
        --backend {{backend}} \
        --num_solutions 30 \
        --output_folder data/schedules/ \
        --table_type {{table_type}} \
        --minimize_mode {{minimize_mode}}

gen-schedules-isolated-tmax:
    just gen-schedule 3A021JEHN02756 cifar-sparse vk isolated tmax
    just gen-schedule 3A021JEHN02756 cifar-dense vk isolated tmax
    just gen-schedule 3A021JEHN02756 tree vk isolated tmax
    just gen-schedule 9b034f1b cifar-sparse vk isolated tmax
    just gen-schedule 9b034f1b cifar-dense vk isolated tmax
    just gen-schedule 9b034f1b tree vk isolated tmax
    just gen-schedule jetson cifar-sparse cu isolated tmax
    just gen-schedule jetson cifar-dense cu isolated tmax
    just gen-schedule jetson tree cu isolated tmax
    just gen-schedule R5CY21Y3VEV cifar-sparse vk isolated tmax
    just gen-schedule R5CY21Y3VEV cifar-dense vk isolated tmax
    just gen-schedule R5CY21Y3VEV tree vk isolated tmax

gen-schedules-isolated-gapness:
    just gen-schedule 3A021JEHN02756 cifar-sparse vk isolated gapness
    just gen-schedule 3A021JEHN02756 cifar-dense vk isolated gapness
    just gen-schedule 3A021JEHN02756 tree vk isolated gapness
    just gen-schedule 9b034f1b cifar-sparse vk isolated gapness
    just gen-schedule 9b034f1b cifar-dense vk isolated gapness
    just gen-schedule 9b034f1b tree vk isolated gapness
    just gen-schedule jetson cifar-sparse cu isolated gapness
    just gen-schedule jetson cifar-dense cu isolated gapness
    just gen-schedule jetson tree cu isolated gapness
    just gen-schedule R5CY21Y3VEV cifar-sparse vk isolated gapness
    just gen-schedule R5CY21Y3VEV cifar-dense vk isolated gapness
    just gen-schedule R5CY21Y3VEV tree vk isolated gapness

gen-schedules-btpm-tmax:
    just gen-schedule 3A021JEHN02756 cifar-sparse vk btpm tmax
    just gen-schedule 3A021JEHN02756 cifar-dense vk btpm tmax
    just gen-schedule 3A021JEHN02756 tree vk btpm tmax
    just gen-schedule 9b034f1b cifar-sparse vk btpm tmax
    just gen-schedule 9b034f1b cifar-dense vk btpm tmax
    just gen-schedule 9b034f1b tree vk btpm tmax
    just gen-schedule jetson cifar-sparse cu btpm tmax
    just gen-schedule jetson cifar-dense cu btpm tmax
    just gen-schedule jetson tree cu btpm tmax
    just gen-schedule R5CY21Y3VEV cifar-sparse vk btpm tmax
    just gen-schedule R5CY21Y3VEV cifar-dense vk btpm tmax
    just gen-schedule R5CY21Y3VEV tree vk btpm tmax

# Good
gen-schedules-btpm-gapness:
    just gen-schedule 3A021JEHN02756 cifar-sparse vk btpm gapness
    just gen-schedule 3A021JEHN02756 cifar-dense vk btpm gapness
    just gen-schedule 3A021JEHN02756 tree vk btpm gapness
    just gen-schedule 9b034f1b cifar-sparse vk btpm gapness
    just gen-schedule 9b034f1b cifar-dense vk btpm gapness
    just gen-schedule 9b034f1b tree vk btpm gapness
    just gen-schedule jetson cifar-sparse cu btpm gapness
    just gen-schedule jetson cifar-dense cu btpm gapness
    just gen-schedule jetson tree cu btpm gapness
    just gen-schedule R5CY21Y3VEV cifar-sparse vk btpm gapness
    just gen-schedule R5CY21Y3VEV cifar-dense vk btpm gapness
    just gen-schedule R5CY21Y3VEV tree vk btpm gapness

# Generate all schedules for all devices, use this, this is fast
gen-schedules-all:
    just gen-schedules-isolated-tmax
    just gen-schedules-isolated-gapness
    just gen-schedules-btpm-tmax
    just gen-schedules-btpm-gapness


# ----------------------------------------------------------------------------
# Auto-Tuning (Optional Step 3)
# For each generated schedule, we will run it 5 times and collect the execution time
# Output execution time is saved under the folder
# 'data/exe_logs/<table_type>_<minimize_mode>/<device>/<app>/<backend>/'
# e.g., data/exe_logs/btpm_gapness/3A021JEHN02756/tree/vk/
# 
# The execution time is saved in the file 'exe_logs_<table_type>_<minimize_mode>_<device>_<app>_<backend>.json'
# e.g., exe_logs_btpm_gapness_3A021JEHN02756_tree_vk.json
# 
# ----------------------------------------------------------------------------

run-schedule device app backend table_type minimize_mode:
    uv run scripts/collect/03_run_schedule.py \
        --log_folder data/exe_logs_{{table_type}}_{{minimize_mode}} \
        --repeat 5 \
        --app {{app}} \
        --backend {{backend}} \
        --device {{device}} \
        --table_type {{table_type}} \
        --minimize_mode {{minimize_mode}} \
        --n-schedules-to-run 30 

run-all-schedule:
    just run-schedule 3A021JEHN02756 cifar-sparse vk btpm gapness
    just run-schedule 3A021JEHN02756 cifar-dense vk btpm gapness
    just run-schedule 3A021JEHN02756 tree vk btpm gapness
    just run-schedule 9b034f1b cifar-sparse vk btpm gapness
    just run-schedule 9b034f1b cifar-dense vk btpm gapness
    just run-schedule 9b034f1b tree vk btpm gapness

run-jetson-schedule:
    just run-schedule jetson cifar-sparse cu btpm gapness
    just run-schedule jetson cifar-dense cu btpm gapness
    just run-schedule jetson tree cu btpm gapness

run-all-schedule-isolated:
    just run-schedule 3A021JEHN02756 cifar-sparse vk isolated tmax
    just run-schedule 3A021JEHN02756 cifar-dense vk isolated tmax
    just run-schedule 3A021JEHN02756 tree vk isolated tmax
    just run-schedule 9b034f1b cifar-sparse vk isolated tmax
    just run-schedule 9b034f1b cifar-dense vk isolated tmax
    just run-schedule 9b034f1b tree vk isolated tmax

# ----------------------------------------------------------------------------
# For result figures
# ----------------------------------------------------------------------------

compare-schedules device app backend n:
    uv run scripts/collect/04_parse_schedules.py -v \
        data/exe_logs_btpm_gapness/{{device}}/{{app}}/{{backend}}/ \
        --schedule-file data/schedules/{{device}}/{{app}}/{{backend}}/schedules_btpm_gapness.json \
        --output tmp1 \
        --max-schedules {{n}}

    uv run scripts/collect/04_parse_schedules.py -v \
        data/exe_logs_isolated_tmax/{{device}}/{{app}}/{{backend}}/ \
        --schedule-file data/schedules/{{device}}/{{app}}/{{backend}}/schedules_isolated_tmax.json \
        --output tmp2 \
        --max-schedules {{n}}

    uv run scripts/collect/04_parse_schedules.py -v \
        data/exe_logs_isolated_gapness/{{device}}/{{app}}/{{backend}}/ \
        --schedule-file data/schedules/{{device}}/{{app}}/{{backend}}/schedules_isolated_gapness.json \
        --output tmp3 \
        --max-schedules {{n}}
    
    uv run scripts/collect/04_parse_schedules.py -v \
        data/exe_logs_btpm_tmax/{{device}}/{{app}}/{{backend}}/ \
        --schedule-file data/schedules/{{device}}/{{app}}/{{backend}}/schedules_btpm_tmax.json \
        --output tmp4 \
        --max-schedules {{n}}


tmp:
    # just run-schedule jetson cifar-sparse cu isolated tmax
    just run-schedule jetson cifar-dense cu isolated tmax
    just run-schedule jetson tree cu isolated tmax