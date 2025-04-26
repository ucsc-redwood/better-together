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
# Final Version
# ----------------------------------------------------------------------------

run-baselines-android:
    xmake r bm-baseline-cifar-dense-vk --device 3A021JEHN02756
    xmake r bm-baseline-cifar-sparse-vk --device 3A021JEHN02756
    xmake r bm-baseline-tree-vk --device 3A021JEHN02756
    xmake r bm-baseline-cifar-dense-vk --device 9b034f1b
    xmake r bm-baseline-cifar-sparse-vk --device 9b034f1b
    xmake r bm-baseline-tree-vk --device 9b034f1b

run-baselines-jetson:
    xmake r bm-baseline-cifar-sparse-cu --device jetson
    xmake r bm-baseline-cifar-dense-cu --device jetson
    xmake r bm-baseline-tree-cu --device jetson
    # xmake r bm-baseline-cifar-sparse-vk --device jetson
    # xmake r bm-baseline-cifar-dense-vk --device jetson
    # xmake r bm-baseline-tree-vk --device jetson

run-baselines-jetsonlowpower:
    xmake r bm-baseline-cifar-sparse-cu --device jetsonlowpower
    xmake r bm-baseline-cifar-dense-cu --device jetsonlowpower
    xmake r bm-baseline-tree-cu --device jetsonlowpower
    # xmake r bm-baseline-cifar-sparse-vk --device jetsonlowpower
    # xmake r bm-baseline-cifar-dense-vk --device jetsonlowpower
    # xmake r bm-baseline-tree-vk --device jetsonlowpower


collect-all-android:
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app tree --backend vk --device 3A021JEHN02756
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app cifar-sparse --backend vk --device 3A021JEHN02756
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app cifar-dense --backend vk --device 3A021JEHN02756
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app tree --backend vk --device 9b034f1b
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app cifar-sparse --backend vk --device 9b034f1b
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app cifar-dense --backend vk --device 9b034f1b

    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app tree --backend vk --device 3A021JEHN02756
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app cifar-sparse --backend vk --device 3A021JEHN02756
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app cifar-dense --backend vk --device 3A021JEHN02756
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app tree --backend vk --device 9b034f1b
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app cifar-sparse --backend vk --device 9b034f1b
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app cifar-dense --backend vk --device 9b034f1b

    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app tree --backend vk --device 3A021JEHN02756
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app cifar-sparse --backend vk --device 3A021JEHN02756
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app cifar-dense --backend vk --device 3A021JEHN02756
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app tree --backend vk --device 9b034f1b
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app cifar-sparse --backend vk --device 9b034f1b
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app cifar-dense --backend vk --device 9b034f1b

collect-all-jetson:
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app cifar-sparse --backend cu --device jetson
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app cifar-dense --backend cu --device jetson
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app tree --backend cu --device jetson

    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app cifar-sparse --backend cu --device jetson
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app cifar-dense --backend cu --device jetson
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app tree --backend cu --device jetson

    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app cifar-sparse --backend cu --device jetson
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app cifar-dense --backend cu --device jetson
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app tree --backend cu --device jetson


collect-all-jetsonlowpower: 
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app cifar-sparse --backend cu --device jetsonlowpower
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app cifar-dense --backend cu --device jetsonlowpower
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app tree --backend cu --device jetsonlowpower

    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app cifar-sparse --backend cu --device jetsonlowpower
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app cifar-dense --backend cu --device jetsonlowpower
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app tree --backend cu --device jetsonlowpower

    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app cifar-sparse --backend cu --device jetsonlowpower
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app cifar-dense --backend cu --device jetsonlowpower
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app tree --backend cu --device jetsonlowpower

only-aggregate:
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --app cifar-sparse --backend vk --device 3A021JEHN02756 --only-aggregate
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --app cifar-dense --backend vk --device 3A021JEHN02756 --only-aggregate

    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --app cifar-sparse --backend vk --device 9b034f1b --only-aggregate
    python3 scripts/collect/00_bm.py --log_folder data/bm_logs --app cifar-dense --backend vk --device 9b034f1b --only-aggregate

make-heatmap:
    python3 scripts/collect/01_make_heatmap.py --log_folder data/bm_logs/  --app cifar-sparse --backend vk --device 3A021JEHN02756 
    python3 scripts/collect/01_make_heatmap.py --log_folder data/bm_logs/  --app cifar-sparse --backend vk --device 9b034f1b 
    python3 scripts/collect/01_make_heatmap.py --log_folder data/bm_logs/  --app cifar-sparse --backend cu --device jetson 
    python3 scripts/collect/01_make_heatmap.py --log_folder data/bm_logs/  --app cifar-sparse --backend cu --device jetsonlowpower 

    python3 scripts/collect/01_make_heatmap.py --log_folder data/bm_logs/  --app cifar-dense --backend vk --device 3A021JEHN02756 --exclude_stages 2,4,8,9
    python3 scripts/collect/01_make_heatmap.py --log_folder data/bm_logs/  --app cifar-dense --backend vk --device 9b034f1b --exclude_stages 2,4,8,9
    python3 scripts/collect/01_make_heatmap.py --log_folder data/bm_logs/  --app cifar-dense --backend cu --device jetson --exclude_stages 2,4,8,9
    python3 scripts/collect/01_make_heatmap.py --log_folder data/bm_logs/  --app cifar-dense --backend cu --device jetsonlowpower --exclude_stages 2,4,8,9

    python3 scripts/collect/01_make_heatmap.py --log_folder data/bm_logs/  --app tree --backend vk --device 3A021JEHN02756
    python3 scripts/collect/01_make_heatmap.py --log_folder data/bm_logs/  --app tree --backend vk --device 9b034f1b
    python3 scripts/collect/01_make_heatmap.py --log_folder data/bm_logs/  --app tree --backend cu --device jetson
    python3 scripts/collect/01_make_heatmap.py --log_folder data/bm_logs/  --app tree --backend cu --device jetsonlowpower


gen-schedules:
    python3 scripts/collect/02_schedule.py --csv_folder data/bm_logs/ --device 3A021JEHN02756 --app cifar-sparse --backend vk --num_solutions 20 --output_folder data/schedules/
    python3 scripts/collect/02_schedule.py --csv_folder data/bm_logs/ --device 3A021JEHN02756 --app cifar-dense --backend vk --num_solutions 20 --output_folder data/schedules/
    python3 scripts/collect/02_schedule.py --csv_folder data/bm_logs/ --device 3A021JEHN02756 --app tree --backend vk --num_solutions 20 --output_folder data/schedules/

    python3 scripts/collect/02_schedule.py --csv_folder data/bm_logs/ --device 9b034f1b --app cifar-sparse --backend vk --num_solutions 20 --output_folder data/schedules/
    python3 scripts/collect/02_schedule.py --csv_folder data/bm_logs/ --device 9b034f1b --app cifar-dense --backend vk --num_solutions 20 --output_folder data/schedules/
    python3 scripts/collect/02_schedule.py --csv_folder data/bm_logs/ --device 9b034f1b --app tree --backend vk --num_solutions 20 --output_folder data/schedules/

    python3 scripts/collect/02_schedule.py --csv_folder data/bm_logs/ --device jetson --app cifar-sparse --backend cu --num_solutions 20 --output_folder data/schedules/
    python3 scripts/collect/02_schedule.py --csv_folder data/bm_logs/ --device jetson --app cifar-dense --backend cu --num_solutions 20 --output_folder data/schedules/
    python3 scripts/collect/02_schedule.py --csv_folder data/bm_logs/ --device jetson --app tree --backend cu --num_solutions 20 --output_folder data/schedules/

    python3 scripts/collect/02_schedule.py --csv_folder data/bm_logs/ --device jetsonlowpower --app cifar-sparse --backend cu --num_solutions 20 --output_folder data/schedules/
    python3 scripts/collect/02_schedule.py --csv_folder data/bm_logs/ --device jetsonlowpower --app cifar-dense --backend cu --num_solutions 20 --output_folder data/schedules/
    python3 scripts/collect/02_schedule.py --csv_folder data/bm_logs/ --device jetsonlowpower --app tree --backend cu --num_solutions 20 --output_folder data/schedules/

gen-schedules-normal:
    python3 scripts/collect/02_schedule_using_normal_table.py --csv_folder data/bm_logs/ --device 3A021JEHN02756 --app cifar-sparse --backend vk --num_solutions 20 --output_folder data/schedules-normal/
    python3 scripts/collect/02_schedule_using_normal_table.py --csv_folder data/bm_logs/ --device 3A021JEHN02756 --app cifar-dense --backend vk --num_solutions 20 --output_folder data/schedules-normal/
    python3 scripts/collect/02_schedule_using_normal_table.py --csv_folder data/bm_logs/ --device 3A021JEHN02756 --app tree --backend vk --num_solutions 20 --output_folder data/schedules-normal/

    python3 scripts/collect/02_schedule_using_normal_table.py --csv_folder data/bm_logs/ --device 9b034f1b --app cifar-sparse --backend vk --num_solutions 20 --output_folder data/schedules-normal/
    python3 scripts/collect/02_schedule_using_normal_table.py --csv_folder data/bm_logs/ --device 9b034f1b --app cifar-dense --backend vk --num_solutions 20 --output_folder data/schedules-normal/
    python3 scripts/collect/02_schedule_using_normal_table.py --csv_folder data/bm_logs/ --device 9b034f1b --app tree --backend vk --num_solutions 20 --output_folder data/schedules-normal/

    python3 scripts/collect/02_schedule_using_normal_table.py --csv_folder data/bm_logs/ --device jetson --app cifar-sparse --backend cu --num_solutions 20 --output_folder data/schedules-normal/
    python3 scripts/collect/02_schedule_using_normal_table.py --csv_folder data/bm_logs/ --device jetson --app cifar-dense --backend cu --num_solutions 20 --output_folder data/schedules-normal/
    python3 scripts/collect/02_schedule_using_normal_table.py --csv_folder data/bm_logs/ --device jetson --app tree --backend cu --num_solutions 20 --output_folder data/schedules-normal/

    python3 scripts/collect/02_schedule_using_normal_table.py --csv_folder data/bm_logs/ --device jetsonlowpower --app cifar-sparse --backend cu --num_solutions 20 --output_folder data/schedules-normal/
    python3 scripts/collect/02_schedule_using_normal_table.py --csv_folder data/bm_logs/ --device jetsonlowpower --app cifar-dense --backend cu --num_solutions 20 --output_folder data/schedules-normal/
    python3 scripts/collect/02_schedule_using_normal_table.py --csv_folder data/bm_logs/ --device jetsonlowpower --app tree --backend cu --num_solutions 20 --output_folder data/schedules-normal/



serve:
    python3 -m http.server --bind 0.0.0.0 --directory data/schedules/ 8080


run-schedule device app backend:
    python3 scripts/collect/03_run_schedule.py \
        --log_folder data/exe_logs \
        --repeat 5 \
        --app {{app}} \
        --backend {{backend}} \
        --device {{device}} \
        --n-schedules-to-run 20

run-schedule-normal device app backend:
    python3 scripts/collect/03_run_schedule.py \
        --log_folder data/exe_logs_normal \
        --repeat 5 \
        --app {{app}} \
        --backend {{backend}} \
        --device {{device}} \
        --use-normal-table True \
        --n-schedules-to-run 20

compare-schedules device app backend:
    python3 scripts/collect/04_parse_schedules_by_widest.py -v \
        data/exe_logs/{{device}}/{{app}}/{{backend}} \
        --model data/schedules/{{device}}/{{app}}/{{backend}}/schedules.json

compare-schedules-adv device app backend:
    python3 scripts/collect/04_parse_schedules_by_widest_advanced.py -v \
        data/exe_logs/{{device}}/{{app}}/{{backend}} \
        --model data/schedules/{{device}}/{{app}}/{{backend}}/schedules.json \
        -o plots/{{device}}/{{app}}/{{backend}}


tmp:
    python3 scripts/collect/04_parse_schedules_by_widest_advanced.py -v \
        data-stable/exe_logs_tmax/3A021JEHN02756/cifar-sparse/vk \
        --model data/schedules/3A021JEHN02756/cifar-sparse/vk/schedules.json \
        -o tmp_dir_tmx


compare-schedules-android:
    just compare-schedules 3A021JEHN02756 cifar-sparse vk
    just compare-schedules 3A021JEHN02756 cifar-dense vk
    just compare-schedules 3A021JEHN02756 tree vk

    # just compare-schedules 9b034f1b cifar-sparse vk
    # just compare-schedules 9b034f1b cifar-dense vk
    # just compare-schedules 9b034f1b tree vk

compare-schedules-android-adv:
    just compare-schedules-adv 3A021JEHN02756 cifar-sparse vk
    just compare-schedules-adv 3A021JEHN02756 cifar-dense vk
    just compare-schedules-adv 3A021JEHN02756 tree vk

    just compare-schedules-adv 9b034f1b cifar-sparse vk
    just compare-schedules-adv 9b034f1b cifar-dense vk
    just compare-schedules-adv 9b034f1b tree vk


make-example-timeline device app backend id:
    python3 scripts/collect/05_timeline.py data/exe_logs/{{device}}/{{app}}/{{backend}}/schedule_run_{{id}}.log \
        --output-dir data/exe_logs/{{device}}/{{app}}/{{backend}}/timeline
