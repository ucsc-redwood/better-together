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

tmp:
    xmake r pipe-ndarray-cu --device jetson --file /home/doremy/Desktop/better-together/data/schedule_files_v3/jetson/CifarDense/schedule_001.json
    xmake r pipe-ndarray-cu --device jetson --file /home/doremy/Desktop/better-together/data/schedule_files_v3/jetson/CifarDense/schedule_002.json
    xmake r pipe-ndarray-cu --device jetson --file /home/doremy/Desktop/better-together/data/schedule_files_v3/jetson/CifarDense/schedule_003.json
    xmake r pipe-ndarray-cu --device jetson --file /home/doremy/Desktop/better-together/data/schedule_files_v3/jetson/CifarDense/schedule_004.json
    xmake r pipe-ndarray-cu --device jetson --file /home/doremy/Desktop/better-together/data/schedule_files_v3/jetson/CifarDense/schedule_005.json
    xmake r pipe-ndarray-cu --device jetson --file /home/doremy/Desktop/better-together/data/schedule_files_v3/jetson/CifarDense/schedule_006.json
    xmake r pipe-ndarray-cu --device jetson --file /home/doremy/Desktop/better-together/data/schedule_files_v3/jetson/CifarDense/schedule_007.json
    xmake r pipe-ndarray-cu --device jetson --file /home/doremy/Desktop/better-together/data/schedule_files_v3/jetson/CifarDense/schedule_008.json
    xmake r pipe-ndarray-cu --device jetson --file /home/doremy/Desktop/better-together/data/schedule_files_v3/jetson/CifarDense/schedule_009.json
    xmake r pipe-ndarray-cu --device jetson --file /home/doremy/Desktop/better-together/data/schedule_files_v3/jetson/CifarDense/schedule_010.json
    xmake r pipe-ndarray-cu --device jetson --file /home/doremy/Desktop/better-together/data/schedule_files_v3/jetson/CifarDense/schedule_011.json
    xmake r pipe-ndarray-cu --device jetson --file /home/doremy/Desktop/better-together/data/schedule_files_v3/jetson/CifarDense/schedule_012.json
    xmake r pipe-ndarray-cu --device jetson --file /home/doremy/Desktop/better-together/data/schedule_files_v3/jetson/CifarDense/schedule_013.json
    xmake r pipe-ndarray-cu --device jetson --file /home/doremy/Desktop/better-together/data/schedule_files_v3/jetson/CifarDense/schedule_014.json
    xmake r pipe-ndarray-cu --device jetson --file /home/doremy/Desktop/better-together/data/schedule_files_v3/jetson/CifarDense/schedule_015.json
    xmake r pipe-ndarray-cu --device jetson --file /home/doremy/Desktop/better-together/data/schedule_files_v3/jetson/CifarDense/schedule_016.json
    xmake r pipe-ndarray-cu --device jetson --file /home/doremy/Desktop/better-together/data/schedule_files_v3/jetson/CifarDense/schedule_017.json
    xmake r pipe-ndarray-cu --device jetson --file /home/doremy/Desktop/better-together/data/schedule_files_v3/jetson/CifarDense/schedule_018.json
    
    
