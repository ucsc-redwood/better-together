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
#  Python Related
#  ----------------------------------------------------------------------------

# using 'uv' to create a virtual environment
venv:
    uv venv .venv
    source .venv/bin/activate.fish

# install dependencies
install-deps:
    uv pip install -r requirements.txt

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
    adb -s ce0717178d7758b00b7e shell "rm -rf /data/local/tmp/*"
    
# List all files in the temporary directory of Android devices
cat-android-tmp:
    adb -s 3A021JEHN02756 shell "ls -la /data/local/tmp"
    adb -s 9b034f1b shell "ls -la /data/local/tmp"
    adb -s ce0717178d7758b00b7e shell "ls -la /data/local/tmp"



# ----------------------------------------------------------------------------
# Used to compare non-full and full, uisng heatmap
# # ----------------------------------------------------------------------------

# # 1) Running the fully BM
# # 2) Running the normal BM
# # 3) Ran multiple times, store the results to a file
# collect-bm-data app:
#     python3 scripts/collect/bm.py --log_folder data/2025-4-17/{{app}}/ \
#         --repeat 3 --target bm-fully-{{app}}-vk

# collect-bm-data-all:
#     just collect-bm-data cifar-sparse

# # 1) From the stored file, make the heatmap
# make-bm-heatmap:
#     python3 scripts/plot/normal_vs_fully_heat.py --folder data/2025-4-17/cifar-sparse/ --exclude_stages 2,4,8,9



# # Generating the schedules (z3)
# # 1) Load the accumulated fully vs. normal BM time
# # 2) Make an averaged BM table
# # 3) Generate the schedules
# # 4) Write the schedules to .json file
# gen-schedules-z3:
#     python3 scripts/gen/schedule.py --csv_path data/2025-4-17/cifar-sparse/3A021JEHN02756_fully.csv \
#         -n 30 --output_file data/2025-4-17/schdules/3A021JEHN02756_cifar-sparse_vk_schedules.json

#     python3 scripts/gen/schedule.py --csv_path data/2025-4-17/cifar-sparse/9b034f1b_fully.csv \
#         -n 30 --output_file data/2025-4-17/schdules/9b034f1b_cifar-sparse_vk_schedules.json


# # Running the schedules
# # 1) Serve the schedules directory
# # So that the android phones can access the schedules via HTTP
# serve:
#     python3 -m http.server --bind 0.0.0.0 --directory data/2025-4-17/schdules/ 8080


# # Run 3 times to get 3 .log files
# # [Input]
# #   schedules.json
# # [Output]
# #   .log files
# run-schedule:
#     python3 runner.py --device 3A021JEHN02756 --app cifar-sparse run -n 30
#     python3 runner.py --device 3A021JEHN02756 --app cifar-sparse run -n 30
#     python3 runner.py --device 3A021JEHN02756 --app cifar-sparse run -n 30

    


#  py parse_schedules_by_widest.py . --model data/2025-4-17/schdules/3A021JEHN02756_cifar-sparse_vk_schedules.json




# # log-to-timeline:
#     python3 scripts/plot/timeline.py  3A021JEHN02756_cifar-sparse_schedules_000.log 


# run-schedule:
#     rm -f 3A021JEHN02756_cifar-dense_schedules.log
#     rm -rf tmp_folder
#     mkdir -p tmp_folder
#     xmake r bm-gen-logs-cifar-dense-vk -l off --device-to-measure 3A021JEHN02756 \
#         --schedule-url http://192.168.1.204:8080/3A021JEHN02756_cifar_dense_vk_schedules.json \
#         --n-schedules-to-run 10 | tee 3A021JEHN02756_cifar-dense_schedules.log

# run-schedule-part-2:
#     python3 scripts/plot/schedule_exe.py --output-dir tmp_folder/ 3A021JEHN02756_cifar-dense_schedules.log > tmp2.txt
#     cat tmp2.txt | grep "Total execution time:" >> accumulated_time.txt
#     cat accumulated_time.txt




# run-schedules-gen-real-time-part-1:
#     rm -f tmp.txt tmp2.txt
#     rm -rf tmp_folder
#     mkdir -p tmp_folder
#     xmake r bm-gen-logs-cifar-sparse-vk > tmp.txt

# run-schedules-gen-real-time-part-2:
#     python3 scripts/plot/schedule_exe.py --output-dir tmp_folder/ tmp.txt > tmp2.txt
#     echo "--------------------------------" >> accumulated_time.txt
#     cat tmp2.txt | grep "Total execution time:" >> accumulated_time.txt
#     cat accumulated_time.txt

# run-schedules-gen-real-time-n-times:
#     echo "1/3..."
#     just run-schedules-gen-real-time-part-1
#     just run-schedules-gen-real-time-part-2
#     echo "2/3..."
#     just run-schedules-gen-real-time-part-1
#     just run-schedules-gen-real-time-part-2
#     echo "3/3..."
#     just run-schedules-gen-real-time-part-1
#     just run-schedules-gen-real-time-part-2

#     python3 parse_times.py --input accumulated_time.txt

# cat-math:
#     python3 scripts/gen_schedule/schedule.py


# # Will need to use the latest schedule
# run-schedules-gen-real-time-part-1:
#     rm -f tmp.txt tmp2.txt
#     rm -rf tmp_folder
#     mkdir -p tmp_folder
#     xmake r bm-gen-logs-cifar-sparse-vk > tmp.txt

# run-schedules-gen-real-time-part-2:
#     python3 scripts/plot/schedule_exe.py --output-dir tmp_folder/ tmp.txt > tmp2.txt
#     echo "--------------------------------" >> accumulated_time.txt
#     cat tmp2.txt | grep "Total execution time:" >> accumulated_time.txt
#     cat accumulated_time.txt

# run-schedules-gen-real-time-n-times:
#     echo "1/3..."
#     just run-schedules-gen-real-time-part-1
#     just run-schedules-gen-real-time-part-2
#     echo "2/3..."
#     just run-schedules-gen-real-time-part-1
#     just run-schedules-gen-real-time-part-2
#     echo "3/3..."
#     just run-schedules-gen-real-time-part-1
#     just run-schedules-gen-real-time-part-2

#     python3 parse_times.py --input accumulated_time.txt

# cat-math:
#     python3 scripts/gen_schedule/schedule.py


# ----------------------------------------------------------------------------
# Final Version
# ----------------------------------------------------------------------------

collect-all-android:
    python3 scripts/collect/bm.py --log_folder data/bm_logs --repeat 1 --app tree --backend vk --device 3A021JEHN02756
    python3 scripts/collect/bm.py --log_folder data/bm_logs --repeat 3 --app cifar-sparse --backend vk --device 3A021JEHN02756
    python3 scripts/collect/bm.py --log_folder data/bm_logs --repeat 3 --app cifar-dense --backend vk --device 3A021JEHN02756

    python3 scripts/collect/bm.py --log_folder data/bm_logs --repeat 1 --app tree --backend vk --device 9b034f1b
    python3 scripts/collect/bm.py --log_folder data/bm_logs --repeat 3 --app cifar-sparse --backend vk --device 9b034f1b
    python3 scripts/collect/bm.py --log_folder data/bm_logs --repeat 3 --app cifar-dense --backend vk --device 9b034f1b


collect-all-jetson:
    python3 scripts/collect/bm.py --log_folder data/bm_logs --repeat 3 --app cifar-sparse --backend cu --device jetson
    python3 scripts/collect/bm.py --log_folder data/bm_logs --repeat 3 --app cifar-dense --backend cu --device jetson

collect-all-jetsonlowpower:
    python3 scripts/collect/bm.py --log_folder data/bm_logs --repeat 3 --app cifar-sparse --backend cu --device jetsonlowpower
    python3 scripts/collect/bm.py --log_folder data/bm_logs --repeat 3 --app cifar-dense --backend cu --device jetsonlowpower

only-aggregate:
    python3 scripts/collect/bm.py --log_folder data/bm_logs --app cifar-sparse --backend vk --device 3A021JEHN02756 --only-aggregate
    python3 scripts/collect/bm.py --log_folder data/bm_logs --app cifar-dense --backend vk --device 3A021JEHN02756 --only-aggregate

    python3 scripts/collect/bm.py --log_folder data/bm_logs --app cifar-sparse --backend vk --device 9b034f1b --only-aggregate
    python3 scripts/collect/bm.py --log_folder data/bm_logs --app cifar-dense --backend vk --device 9b034f1b --only-aggregate

make-heatmap:
    python3 scripts/collect/make_heatmap.py --log_folder data/bm_logs/  --app cifar-sparse --backend vk --device 3A021JEHN02756 
    python3 scripts/collect/make_heatmap.py --log_folder data/bm_logs/  --app cifar-sparse --backend vk --device 9b034f1b 
    python3 scripts/collect/make_heatmap.py --log_folder data/bm_logs/  --app cifar-sparse --backend cu --device jetson 
    python3 scripts/collect/make_heatmap.py --log_folder data/bm_logs/  --app cifar-sparse --backend cu --device jetsonlowpower 

    python3 scripts/collect/make_heatmap.py --log_folder data/bm_logs/  --app cifar-dense --backend vk --device 3A021JEHN02756 --exclude_stages 2,4,8,9
    python3 scripts/collect/make_heatmap.py --log_folder data/bm_logs/  --app cifar-dense --backend vk --device 9b034f1b --exclude_stages 2,4,8,9
    python3 scripts/collect/make_heatmap.py --log_folder data/bm_logs/  --app cifar-dense --backend cu --device jetson --exclude_stages 2,4,8,9
    python3 scripts/collect/make_heatmap.py --log_folder data/bm_logs/  --app cifar-dense --backend cu --device jetsonlowpower --exclude_stages 2,4,8,9

    python3 scripts/collect/make_heatmap.py --log_folder data/bm_logs/  --app tree --backend vk --device 3A021JEHN02756
    python3 scripts/collect/make_heatmap.py --log_folder data/bm_logs/  --app tree --backend vk --device 9b034f1b
    python3 scripts/collect/make_heatmap.py --log_folder data/bm_logs/  --app tree --backend cu --device jetson
    python3 scripts/collect/make_heatmap.py --log_folder data/bm_logs/  --app tree --backend cu --device jetsonlowpower


gen-schedules:
    python3 scripts/gen/schedule.py --csv_folder data/bm_logs/ --device 3A021JEHN02756 --app cifar-sparse --backend vk --num_solutions 20 --output_folder data/schedules/
    python3 scripts/gen/schedule.py --csv_folder data/bm_logs/ --device 3A021JEHN02756 --app cifar-dense --backend vk --num_solutions 20 --output_folder data/schedules/
    python3 scripts/gen/schedule.py --csv_folder data/bm_logs/ --device 3A021JEHN02756 --app tree --backend vk --num_solutions 20 --output_folder data/schedules/

    python3 scripts/gen/schedule.py --csv_folder data/bm_logs/ --device 9b034f1b --app cifar-sparse --backend vk --num_solutions 20 --output_folder data/schedules/
    python3 scripts/gen/schedule.py --csv_folder data/bm_logs/ --device 9b034f1b --app cifar-dense --backend vk --num_solutions 20 --output_folder data/schedules/
    python3 scripts/gen/schedule.py --csv_folder data/bm_logs/ --device 9b034f1b --app tree --backend vk --num_solutions 20 --output_folder data/schedules/

    python3 scripts/gen/schedule.py --csv_folder data/bm_logs/ --device jetson --app cifar-sparse --backend cu --num_solutions 20 --output_folder data/schedules/
    python3 scripts/gen/schedule.py --csv_folder data/bm_logs/ --device jetsonlowpower --app cifar-sparse --backend cu --num_solutions 20 --output_folder data/schedules/

    python3 scripts/gen/schedule.py --csv_folder data/bm_logs/ --device jetson --app cifar-dense --backend cu --num_solutions 20 --output_folder data/schedules/
    python3 scripts/gen/schedule.py --csv_folder data/bm_logs/ --device jetsonlowpower --app cifar-dense --backend cu --num_solutions 20 --output_folder data/schedules/


serve:
    python3 -m http.server --bind 0.0.0.0 --directory data/schedules/ 8080


run-schedule:
    xmake r bm-gen-logs-cifar-sparse-vk --device 3A021JEHN02756 --schedule-url http://192.168.1.204:8080/3A021JEHN02756_cifar-sparse_vk_fully_schedules.json --n-schedules-to-run 10
    xmake r bm-gen-logs-cifar-sparse-vk --device 9b034f1b --schedule-url http://192.168.1.204:8080/9b034f1b_cifar-sparse_vk_fully_schedules.json --n-schedules-to-run 10

    # xmake r bm-gen-logs-cifar-dense-vk --device 3A021JEHN02756 --schedule-url http://192.168.1.204:8080/3A021JEHN02756_cifar-dense_vk_fully_schedules.json --n-schedules-to-run 10  
    # xmake r bm-gen-logs-cifar-dense-vk --device 9b034f1b --schedule-url http://192.168.1.204:8080/9b034f1b_cifar-dense_vk_fully_schedules.json --n-schedules-to-run 10

