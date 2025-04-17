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
# Tmp
# ----------------------------------------------------------------------------




# # This is used to compute the real measurement of the pipeline schedule
# run-best-benchmarks:
#     rm BM_best_txt
#     touch BM_best_txt
#     xmake r bm-best-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_1/ | tee -a BM_best_txt
#     xmake r bm-best-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_2/ | tee -a BM_best_txt
#     xmake r bm-best-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_3/ | tee -a BM_best_txt
#     xmake r bm-best-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_4/ | tee -a BM_best_txt
#     xmake r bm-best-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_5/ | tee -a BM_best_txt
#     xmake r bm-best-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_6/ | tee -a BM_best_txt
#     xmake r bm-best-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_7/ | tee -a BM_best_txt
#     xmake r bm-best-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_8/ | tee -a BM_best_txt
#     xmake r bm-best-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_9/ | tee -a BM_best_txt
#     xmake r bm-best-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_10/ | tee -a BM_best_txt

# # # This is used to generate benchmark Table, (used to generate math model)
# # run-best-benchmarks-table:
# #     rm -f BM_benchmark_table.txt
# #     touch BM_benchmark_table.txt
# #     xmake r bm-pipe-cifar-sparse-vk -l off --benchmark_filter=VK/CifarSparse/Baseline/iterations:1/ | tee -a BM_benchmark_table.txt
# #     xmake r bm-pipe-cifar-sparse-vk -l off --benchmark_filter=VK/CifarSparse/Baseline/iterations:2/ | tee -a BM_benchmark_table.txt
# #     xmake r bm-pipe-cifar-sparse-vk -l off --benchmark_filter=VK/CifarSparse/Baseline/iterations:3/ | tee -a BM_benchmark_table.txt
# #     xmake r bm-pipe-cifar-sparse-vk -l off --benchmark_filter=VK/CifarSparse/Baseline/iterations:4/ | tee -a BM_benchmark_table.txt
# #     xmake r bm-pipe-cifar-sparse-vk -l off --benchmark_filter=VK/CifarSparse/Baseline/iterations:5/ | tee -a BM_benchmark_table.txt
# #     xmake r bm-pipe-cifar-sparse-vk -l off --benchmark_filter=VK/CifarSparse/Baseline/iterations:6/ | tee -a BM_benchmark_table.txt
# #     xmake r bm-pipe-cifar-sparse-vk -l off --benchmark_filter=VK/CifarSparse/Baseline/iterations:7/ | tee -a BM_benchmark_table.txt
# #     xmake r bm-pipe-cifar-sparse-vk -l off --benchmark_filter=VK/CifarSparse/Baseline/iterations:8/ | tee -a BM_benchmark_table.txt
# #     xmake r bm-pipe-cifar-sparse-vk -l off --benchmark_filter=VK/CifarSparse/Baseline/iterations:9/ | tee -a BM_benchmark_table.txt
# #     xmake r bm-pipe-cifar-sparse-vk -l off --benchmark_filter=VK/CifarSparse/Baseline/iterations:10/ | tee -a BM_benchmark_table.txt

# # -----------------------------------------------------------------------------
# # Target 2: Process the raw output using sed
# # This extracts lines 8 to 1707 and removes the unwanted "Chunk 3" blocks with zero times
# # -----------------------------------------------------------------------------
# process-results:
# 	@echo "Processing raw benchmark outputs..."
# 	sed -n '11,1710p' BM_pipe_cifar_sparse_vk_schedule_1.txt | sed '/^  Chunk 3:$/{N;N;N;/^  Chunk 3:\n    Start: 0 us\n    End: 0 us\n    Duration: 0 us$/d}' > BM_pipe_cifar_sparse_vk_schedule_1.txt
# 	sed -n '11,1710p' BM_pipe_cifar_sparse_vk_schedule_2.txt | sed '/^  Chunk 3:$/{N;N;N;/^  Chunk 3:\n    Start: 0 us\n    End: 0 us\n    Duration: 0 us$/d}' > BM_pipe_cifar_sparse_vk_schedule_2.txt
# 	sed -n '11,1710p' BM_pipe_cifar_sparse_vk_schedule_3.txt | sed '/^  Chunk 3:$/{N;N;N;/^  Chunk 3:\n    Start: 0 us\n    End: 0 us\n    Duration: 0 us$/d}' > BM_pipe_cifar_sparse_vk_schedule_3.txt
# 	sed -n '11,1710p' BM_pipe_cifar_sparse_vk_schedule_4.txt | sed '/^  Chunk 3:$/{N;N;N;/^  Chunk 3:\n    Start: 0 us\n    End: 0 us\n    Duration: 0 us$/d}' > BM_pipe_cifar_sparse_vk_schedule_4.txt
# 	sed -n '11,1710p' BM_pipe_cifar_sparse_vk_schedule_5.txt | sed '/^  Chunk 3:$/{N;N;N;/^  Chunk 3:\n    Start: 0 us\n    End: 0 us\n    Duration: 0 us$/d}' > BM_pipe_cifar_sparse_vk_schedule_5.txt
# 	sed -n '11,1710p' BM_pipe_cifar_sparse_vk_schedule_6.txt | sed '/^  Chunk 3:$/{N;N;N;/^  Chunk 3:\n    Start: 0 us\n    End: 0 us\n    Duration: 0 us$/d}' > BM_pipe_cifar_sparse_vk_schedule_6.txt
# 	sed -n '11,1710p' BM_pipe_cifar_sparse_vk_schedule_7.txt | sed '/^  Chunk 3:$/{N;N;N;/^  Chunk 3:\n    Start: 0 us\n    End: 0 us\n    Duration: 0 us$/d}' > BM_pipe_cifar_sparse_vk_schedule_7.txt
# 	sed -n '11,1710p' BM_pipe_cifar_sparse_vk_schedule_8.txt | sed '/^  Chunk 3:$/{N;N;N;/^  Chunk 3:\n    Start: 0 us\n    End: 0 us\n    Duration: 0 us$/d}' > BM_pipe_cifar_sparse_vk_schedule_8.txt
# 	sed -n '11,1710p' BM_pipe_cifar_sparse_vk_schedule_9.txt | sed '/^  Chunk 3:$/{N;N;N;/^  Chunk 3:\n    Start: 0 us\n    End: 0 us\n    Duration: 0 us$/d}' > BM_pipe_cifar_sparse_vk_schedule_9.txt
# 	sed -n '11,1710p' BM_pipe_cifar_sparse_vk_schedule_10.txt | sed '/^  Chunk 3:$/{N;N;N;/^  Chunk 3:\n    Start: 0 us\n    End: 0 us\n    Duration: 0 us$/d}' > BM_pipe_cifar_sparse_vk_schedule_10.txt


# # -----------------------------------------------------------------------------
# # Target 3: Generate figures from processed data using Python.
# # -----------------------------------------------------------------------------
# make-fig:
# 	@echo "Generating figures..."
# 	python3 scripts-v2/analysis/pipe.py \
# 		scripts-v2/analysis/BM_pipe_cifar_sparse_vk_schedule_1.txt \
# 		-o task_execution_timeline_wide_cifar_sparse_vk_schedule_1 \
# 		--task-start 25 --task-end 50
# 	python3 scripts-v2/analysis/pipe.py \
# 		scripts-v2/analysis/BM_pipe_cifar_sparse_vk_schedule_2.txt \
# 		-o task_execution_timeline_wide_cifar_sparse_vk_schedule_2 \
# 		--task-start 25 --task-end 50
# 	python3 scripts-v2/analysis/pipe.py \
# 		scripts-v2/analysis/BM_pipe_cifar_sparse_vk_schedule_3.txt \
# 		-o task_execution_timeline_wide_cifar_sparse_vk_schedule_3 \
# 		--task-start 25 --task-end 50
# 	python3 scripts-v2/analysis/pipe.py \
# 		scripts-v2/analysis/BM_pipe_cifar_sparse_vk_schedule_4.txt \
# 		-o task_execution_timeline_wide_cifar_sparse_vk_schedule_4 \
# 		--task-start 25 --task-end 50
# 	python3 scripts-v2/analysis/pipe.py \
# 		scripts-v2/analysis/BM_pipe_cifar_sparse_vk_schedule_5.txt \
# 		-o task_execution_timeline_wide_cifar_sparse_vk_schedule_5 \
# 		--task-start 25 --task-end 50
# 	python3 scripts-v2/analysis/pipe.py \
# 		scripts-v2/analysis/BM_pipe_cifar_sparse_vk_schedule_6.txt \
# 		-o task_execution_timeline_wide_cifar_sparse_vk_schedule_6 \
# 		--task-start 25 --task-end 50
# 	python3 scripts-v2/analysis/pipe.py \
# 		scripts-v2/analysis/BM_pipe_cifar_sparse_vk_schedule_7.txt \
# 		-o task_execution_timeline_wide_cifar_sparse_vk_schedule_7 \
# 		--task-start 25 --task-end 50
# 	python3 scripts-v2/analysis/pipe.py \
# 		scripts-v2/analysis/BM_pipe_cifar_sparse_vk_schedule_8.txt \
# 		-o task_execution_timeline_wide_cifar_sparse_vk_schedule_8 \
# 		--task-start 25 --task-end 50
# 	python3 scripts-v2/analysis/pipe.py \
# 		scripts-v2/analysis/BM_pipe_cifar_sparse_vk_schedule_9.txt \
# 		-o task_execution_timeline_wide_cifar_sparse_vk_schedule_9 \
# 		--task-start 25 --task-end 50
# 	python3 scripts-v2/analysis/pipe.py \
# 		scripts-v2/analysis/BM_pipe_cifar_sparse_vk_schedule_10.txt \
# 		-o task_execution_timeline_wide_cifar_sparse_vk_schedule_10 \
# 		--task-start 25 --task-end 50

# # -----------------------------------------------------------------------------
# # "all" target to run every step in sequence.
# # -----------------------------------------------------------------------------
# all: 
#     just run-benchmarks 
#     just process-results 
#     just make-fig


# compare-full-and-non-full stage:
#     # Run non-full benchmark
#     xmake r bm-real-cifar-sparse-vk --stage {{stage}} -l off | tee non_full_stage_{{stage}}.txt

#     # Run full benchmark
#     xmake r bm-real-cifar-sparse-vk --stage {{stage}} -l off --full | tee full_stage_{{stage}}.txt

#     # Extract and compare AVG metrics
#     @echo "\n====== COMPARISON OF AVG METRICS ======"
#     @echo "Processor | Non-Full (ms) | Full (ms)"
#     @echo "--------------------------------------"
#     @awk -F'|' '/PROCESSOR=/{split($1,p,"="); split($4,a,"="); printf "%s: %s ms\n", p[2], a[2]}' non_full_stage_{{stage}}.txt | sort
#     @echo "--------------------------------------"
#     @awk -F'|' '/PROCESSOR=/{split($1,p,"="); split($4,a,"="); printf "%s: %s ms\n", p[2], a[2]}' full_stage_{{stage}}.txt | sort

# compare-all:
#     just compare-full-and-non-full 1
#     just compare-full-and-non-full 2
#     just compare-full-and-non-full 3
#     just compare-full-and-non-full 4
#     just compare-full-and-non-full 5
#     just compare-full-and-non-full 6
#     just compare-full-and-non-full 7
#     just compare-full-and-non-full 8
#     just compare-full-and-non-full 9



# ----------------------------------------------------------------------------
# Measure Cifar-Sparse Real Time
# # ----------------------------------------------------------------------------

# try:
#     xmake r bm-schedule-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_0/ | tee BM_pipe_cifar_sparse_vk_schedule_0.txt
#     xmake r bm-schedule-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_1/ | tee BM_pipe_cifar_sparse_vk_schedule_1.txt
#     xmake r bm-schedule-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_2/ | tee BM_pipe_cifar_sparse_vk_schedule_2.txt
#     xmake r bm-schedule-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_3/ | tee BM_pipe_cifar_sparse_vk_schedule_3.txt
#     xmake r bm-schedule-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_4/ | tee BM_pipe_cifar_sparse_vk_schedule_4.txt
#     xmake r bm-schedule-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_5/ | tee BM_pipe_cifar_sparse_vk_schedule_5.txt
#     xmake r bm-schedule-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_6/ | tee BM_pipe_cifar_sparse_vk_schedule_6.txt
#     xmake r bm-schedule-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_7/ | tee BM_pipe_cifar_sparse_vk_schedule_7.txt
#     xmake r bm-schedule-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_8/ | tee BM_pipe_cifar_sparse_vk_schedule_8.txt
#     xmake r bm-schedule-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_9/ | tee BM_pipe_cifar_sparse_vk_schedule_9.txt
#     xmake r bm-schedule-cifar-sparse-vk -l off --benchmark_filter=BM_pipe_cifar_sparse_vk_schedule_10/ | tee BM_pipe_cifar_sparse_vk_schedule_10.txt


# ----------------------------------------------------------------------------
# Measure Cifar-Sparse Real Time and Gen Figure
# ----------------------------------------------------------------------------

# py scripts-v2/gen_schedule/schedule.py

# make_bm_log stage:
#     xmake r gen-records-cifar-sparse-vk --schedule {{stage}} | sed -n '11,1710p' | tee BM_best_raw_stage_{{stage}}.txt

# log_to_figure stage:
#     python3 scripts-v2/analysis/gen_chunk_figure.py BM_best_raw_stage_{{stage}}.txt --output BM_best_raw_stage_{{stage}}.png --start-time 0 --end-time 1

# try-all:
#     just make_bm_log 1
#     just make_bm_log 2
#     just make_bm_log 3
#     just make_bm_log 4
#     just make_bm_log 5
#     just make_bm_log 6
#     just make_bm_log 7
#     just make_bm_log 8
#     just make_bm_log 9
#     just make_bm_log 10


# # Average task durations for each schedule:
# # Schedule 0: 12.2012 ms
# # Schedule 1: 12.416778 ms
# # Schedule 2: 11.824737 ms
# # Schedule 3: 7.610159 ms
# # Schedule 4: 9.040559 ms
# # Schedule 5: 12.960554 ms
# # Schedule 6: 7.108262 ms
# # Schedule 7: 12.148757 ms
# # Schedule 8: 6.685347 ms
# # Schedule 9: 7.954996 ms
# # Schedule 10: 12.057925 ms
# try-all-figure:
#     just log_to_figure 1 | tee BM_best_raw_schedule_1_analysis.txt
#     just log_to_figure 2 | tee BM_best_raw_schedule_2_analysis.txt
#     just log_to_figure 3 | tee BM_best_raw_schedule_3_analysis.txt
#     just log_to_figure 4 | tee BM_best_raw_schedule_4_analysis.txt
#     just log_to_figure 5 | tee BM_best_raw_schedule_5_analysis.txt
#     just log_to_figure 6 | tee BM_best_raw_schedule_6_analysis.txt
#     just log_to_figure 7 | tee BM_best_raw_schedule_7_analysis.txt
#     just log_to_figure 8 | tee BM_best_raw_schedule_8_analysis.txt
#     just log_to_figure 9 | tee BM_best_raw_schedule_9_analysis.txt
#     just log_to_figure 10 | tee BM_best_raw_schedule_10_analysis.txt
#     cat BM_best_raw_schedule_*_analysis.txt | grep "All Tasks Average:" | awk -F'[()]' '{print $2}' | awk '{
#         if (NR%4==1) max=$1
#         if ($1>max) max=$1
#         if (NR%4==0) print max
#     }'



# # This is ued to generate pipeline graph
# run-benchmarks-cifar-sparse-vk stage:
#     xmake r bm-table-cifar-sparse-vk --stage {{stage}} -l off | tee non_full_stage_{{stage}}.txt
#     xmake r bm-table-cifar-sparse-vk --stage {{stage}} -l off --full | tee full_stage_{{stage}}.txt

#     # Extract and compare AVG metrics
#     @echo "\n====== COMPARISON OF AVG METRICS ======"
#     @echo "Processor | Non-Full (ms) | Full (ms)"
#     @echo "--------------------------------------"
#     @awk -F'|' '/PROCESSOR=/{split($1,p,"="); split($4,a,"="); printf "%s: %s ms\n", p[2], a[2]}' non_full_stage_{{stage}}.txt
#     @echo "--------------------------------------"
#     @awk -F'|' '/PROCESSOR=/{split($1,p,"="); split($4,a,"="); printf "%s: %s ms\n", p[2], a[2]}' full_stage_{{stage}}.txt


# ----------------------------------------------------------------------------
# Use this to generate BM table for Android devices (full and non-full)
# ----------------------------------------------------------------------------

# run-benchmarks-vk-full app device:
#     rm -f BM_table_{{app}}_vk_{{device}}_full.txt
#     rm -f BM_table_{{app}}_vk_{{device}}_full.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 1 -l off --full --device-to-measure {{device}} | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_{{device}}_full.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 2 -l off --full --device-to-measure {{device}} | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_{{device}}_full.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 3 -l off --full --device-to-measure {{device}} | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_{{device}}_full.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 4 -l off --full --device-to-measure {{device}} | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_{{device}}_full.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 5 -l off --full --device-to-measure {{device}} | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_{{device}}_full.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 6 -l off --full --device-to-measure {{device}} | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_{{device}}_full.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 7 -l off --full --device-to-measure {{device}} | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_{{device}}_full.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 8 -l off --full --device-to-measure {{device}} | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_{{device}}_full.txt.tmp  
#     xmake r bm-table-{{app}}-vk --stage 9 -l off --full --device-to-measure {{device}} | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_{{device}}_full.txt.tmp

#     awk -F'|' '{p=a=min=max="";for(i=1;i<=NF;i++){if($i~/^PROCESSOR=/)p=$i;if($i~/^AVG=/)a=$i;if($i~/^MIN=/)min=$i;if($i~/^MAX=/)max=$i}print p"|"a"|"min"|"max; if(NR%4==0)print ""}' BM_table_{{app}}_vk_{{device}}_full.txt.tmp > BM_table_{{app}}_vk_{{device}}_full.txt
#     cat BM_table_{{app}}_vk_{{device}}_full.txt

# run-benchmarks-vk app device:
#     rm -f BM_table_{{app}}_vk_{{device}}.txt
#     rm -f BM_table_{{app}}_vk_{{device}}.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 1 -l off --device-to-measure {{device}} | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_{{device}}.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 2 -l off --device-to-measure {{device}} | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_{{device}}.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 3 -l off --device-to-measure {{device}} | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_{{device}}.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 4 -l off --device-to-measure {{device}} | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_{{device}}.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 5 -l off --device-to-measure {{device}} | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_{{device}}.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 6 -l off --device-to-measure {{device}} | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_{{device}}.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 7 -l off --device-to-measure {{device}} | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_{{device}}.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 8 -l off --device-to-measure {{device}} | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_{{device}}.txt.tmp  
#     xmake r bm-table-{{app}}-vk --stage 9 -l off --device-to-measure {{device}} | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_{{device}}.txt.tmp

#     awk -F'|' '{p=a=min=max="";for(i=1;i<=NF;i++){if($i~/^PROCESSOR=/)p=$i;if($i~/^AVG=/)a=$i;if($i~/^MIN=/)min=$i;if($i~/^MAX=/)max=$i}print p"|"a"|"min"|"max; if(NR%4==0)print ""}' BM_table_{{app}}_vk_{{device}}.txt.tmp > BM_table_{{app}}_vk_{{device}}.txt
#     cat BM_table_{{app}}_vk_{{device}}.txt

# run-benchmarks-vk-all app:
#     just run-benchmarks-vk-full {{app}} 3A021JEHN02756
#     just run-benchmarks-vk {{app}} 3A021JEHN02756
#     just run-benchmarks-vk-full {{app}} 9b034f1b
#     just run-benchmarks-vk {{app}} 9b034f1b


# # ----------------------------------------------------------------------------
# # Use this to generate BM table for Jetson devices (full and non-full)
# # ----------------------------------------------------------------------------

# run-benchmarks-vk-jetson-full app:
#     rm -f BM_table_{{app}}_vk_jetson_full.txt
#     rm -f BM_table_{{app}}_vk_jetson_full.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 1 --device jetson --device-to-measure jetson --full | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_jetson_full.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 2 --device jetson --device-to-measure jetson --full | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_jetson_full.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 3 --device jetson --device-to-measure jetson --full | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_jetson_full.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 4 --device jetson --device-to-measure jetson --full | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_jetson_full.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 5 --device jetson --device-to-measure jetson --full | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_jetson_full.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 6 --device jetson --device-to-measure jetson --full | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_jetson_full.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 7 --device jetson --device-to-measure jetson --full | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_jetson_full.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 8 --device jetson --device-to-measure jetson --full | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_jetson_full.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 9 --device jetson --device-to-measure jetson --full | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_jetson_full.txt.tmp

#     awk -F'|' '{p=a=min=max="";for(i=1;i<=NF;i++){if($i~/^PROCESSOR=/)p=$i;if($i~/^AVG=/)a=$i;if($i~/^MIN=/)min=$i;if($i~/^MAX=/)max=$i}print p"|"a"|"min"|"max; if(NR%2==0)print ""}' BM_table_{{app}}_vk_jetson_full.txt.tmp > BM_table_{{app}}_vk_jetson_full.txt
#     cat BM_table_{{app}}_vk_jetson_full.txt

# run-benchmarks-vk-jetson app:
#     rm -f BM_table_{{app}}_vk_jetson.txt
#     rm -f BM_table_{{app}}_vk_jetson.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 1 --device jetson --device-to-measure jetson  | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_jetson.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 2 --device jetson --device-to-measure jetson  | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_jetson.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 3 --device jetson --device-to-measure jetson  | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_jetson.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 4 --device jetson --device-to-measure jetson  | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_jetson.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 5 --device jetson --device-to-measure jetson  | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_jetson.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 6 --device jetson --device-to-measure jetson  | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_jetson.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 7 --device jetson --device-to-measure jetson  | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_jetson.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 8 --device jetson --device-to-measure jetson  | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_jetson.txt.tmp
#     xmake r bm-table-{{app}}-vk --stage 9 --device jetson --device-to-measure jetson  | grep "PROCESSOR=" | tee -a BM_table_{{app}}_vk_jetson.txt.tmp

#     awk -F'|' '{p=a=min=max="";for(i=1;i<=NF;i++){if($i~/^PROCESSOR=/)p=$i;if($i~/^AVG=/)a=$i;if($i~/^MIN=/)min=$i;if($i~/^MAX=/)max=$i}print p"|"a"|"min"|"max; if(NR%2==0)print ""}' BM_table_{{app}}_vk_jetson.txt.tmp > BM_table_{{app}}_vk_jetson.txt
#     cat BM_table_{{app}}_vk_jetson.txt

# run-benchmarks-vk-jetson-all app:
#     just run-benchmarks-vk-jetson-full {{app}}
#     just run-benchmarks-vk-jetson {{app}}



# # This is ued to generate Execution Log
# gen-log:
#     xmake r bm-gen-logs-cifar-sparse-vk --schedule 1 | tee BM_pipe_cifar_sparse_vk_schedule_1.txt.tmp
#     xmake r bm-gen-logs-cifar-sparse-vk --schedule 2 | tee BM_pipe_cifar_sparse_vk_schedule_2.txt.tmp
#     xmake r bm-gen-logs-cifar-sparse-vk --schedule 3 | tee BM_pipe_cifar_sparse_vk_schedule_3.txt.tmp
#     xmake r bm-gen-logs-cifar-sparse-vk --schedule 4 | tee BM_pipe_cifar_sparse_vk_schedule_4.txt.tmp
#     xmake r bm-gen-logs-cifar-sparse-vk --schedule 5 | tee BM_pipe_cifar_sparse_vk_schedule_5.txt.tmp
#     xmake r bm-gen-logs-cifar-sparse-vk --schedule 6 | tee BM_pipe_cifar_sparse_vk_schedule_6.txt.tmp
#     xmake r bm-gen-logs-cifar-sparse-vk --schedule 7 | tee BM_pipe_cifar_sparse_vk_schedule_7.txt.tmp
#     xmake r bm-gen-logs-cifar-sparse-vk --schedule 8 | tee BM_pipe_cifar_sparse_vk_schedule_8.txt.tmp
#     xmake r bm-gen-logs-cifar-sparse-vk --schedule 9 | tee BM_pipe_cifar_sparse_vk_schedule_9.txt.tmp
#     xmake r bm-gen-logs-cifar-sparse-vk --schedule 10 | tee BM_pipe_cifar_sparse_vk_schedule_10.txt.tmp
#     sed -n '11,1710p' BM_pipe_cifar_sparse_vk_schedule_1.txt.tmp > BM_pipe_cifar_sparse_vk_schedule_1.txt
#     sed -n '11,1710p' BM_pipe_cifar_sparse_vk_schedule_2.txt.tmp > BM_pipe_cifar_sparse_vk_schedule_2.txt
#     sed -n '11,1710p' BM_pipe_cifar_sparse_vk_schedule_3.txt.tmp > BM_pipe_cifar_sparse_vk_schedule_3.txt
#     sed -n '11,1710p' BM_pipe_cifar_sparse_vk_schedule_4.txt.tmp > BM_pipe_cifar_sparse_vk_schedule_4.txt
#     sed -n '11,1710p' BM_pipe_cifar_sparse_vk_schedule_5.txt.tmp > BM_pipe_cifar_sparse_vk_schedule_5.txt
#     sed -n '11,1710p' BM_pipe_cifar_sparse_vk_schedule_6.txt.tmp > BM_pipe_cifar_sparse_vk_schedule_6.txt
#     sed -n '11,1710p' BM_pipe_cifar_sparse_vk_schedule_7.txt.tmp > BM_pipe_cifar_sparse_vk_schedule_7.txt
#     sed -n '11,1710p' BM_pipe_cifar_sparse_vk_schedule_8.txt.tmp > BM_pipe_cifar_sparse_vk_schedule_8.txt
#     sed -n '11,1710p' BM_pipe_cifar_sparse_vk_schedule_9.txt.tmp > BM_pipe_cifar_sparse_vk_schedule_9.txt
#     sed -n '11,1710p' BM_pipe_cifar_sparse_vk_schedule_10.txt.tmp > BM_pipe_cifar_sparse_vk_schedule_10.txt

#     rm BM_pipe_cifar_sparse_vk_schedule_*.txt.tmp

# gen-log-make-fig:
#     python3 scripts-v2/analysis/gen_chunk_figure.py BM_pipe_cifar_sparse_vk_schedule_1.txt --output BM_pipe_cifar_sparse_vk_schedule_1.png --start-time 0 --end-time 1
#     python3 scripts-v2/analysis/gen_chunk_figure.py BM_pipe_cifar_sparse_vk_schedule_2.txt --output BM_pipe_cifar_sparse_vk_schedule_2.png --start-time 0 --end-time 1
#     python3 scripts-v2/analysis/gen_chunk_figure.py BM_pipe_cifar_sparse_vk_schedule_3.txt --output BM_pipe_cifar_sparse_vk_schedule_3.png --start-time 0 --end-time 1
#     python3 scripts-v2/analysis/gen_chunk_figure.py BM_pipe_cifar_sparse_vk_schedule_4.txt --output BM_pipe_cifar_sparse_vk_schedule_4.png --start-time 0 --end-time 1
#     python3 scripts-v2/analysis/gen_chunk_figure.py BM_pipe_cifar_sparse_vk_schedule_5.txt --output BM_pipe_cifar_sparse_vk_schedule_5.png --start-time 0 --end-time 1
#     python3 scripts-v2/analysis/gen_chunk_figure.py BM_pipe_cifar_sparse_vk_schedule_6.txt --output BM_pipe_cifar_sparse_vk_schedule_6.png --start-time 0 --end-time 1
#     python3 scripts-v2/analysis/gen_chunk_figure.py BM_pipe_cifar_sparse_vk_schedule_7.txt --output BM_pipe_cifar_sparse_vk_schedule_7.png --start-time 0 --end-time 1
#     python3 scripts-v2/analysis/gen_chunk_figure.py BM_pipe_cifar_sparse_vk_schedule_8.txt --output BM_pipe_cifar_sparse_vk_schedule_8.png --start-time 0 --end-time 1
#     python3 scripts-v2/analysis/gen_chunk_figure.py BM_pipe_cifar_sparse_vk_schedule_9.txt --output BM_pipe_cifar_sparse_vk_schedule_9.png --start-time 0 --end-time 1
#     python3 scripts-v2/analysis/gen_chunk_figure.py BM_pipe_cifar_sparse_vk_schedule_10.txt --output BM_pipe_cifar_sparse_vk_schedule_10.png --start-time 0 --end-time 1



# ----------------------------------------------------------------------------
# Used to compare non-full and full, uisng heatmap
# ----------------------------------------------------------------------------

collect-bm-data app:
    python3 scripts/collect/bm.py --log_folder data/2025-4-16/{{app}}/ \
        --repeat 3 --target bm-fully-{{app}}-vk

collect-bm-data-all:
    just collect-bm-data cifar-sparse
    just collect-bm-data cifar-dense

make-bm-heatmap:
    python3 scripts/plot/normal_vs_fully_heat.py --folder data/2025-4-16/cifar-sparse/ --exclude_stages 2,4,8,9
    python3 scripts/plot/normal_vs_fully_heat.py --folder data/2025-4-16/cifar-dense/ --exclude_stages 2,4,8,9


# Will need to use the latest schedule
run-schedules-gen-real-time-part-1:
    rm -f tmp.txt tmp2.txt
    rm -rf tmp_folder
    mkdir -p tmp_folder
    xmake r bm-gen-logs-cifar-sparse-vk > tmp.txt

run-schedules-gen-real-time-part-2:
    python3 scripts/plot/schedule_exe.py --output-dir tmp_folder/ tmp.txt > tmp2.txt
    echo "--------------------------------" >> accumulated_time.txt
    cat tmp2.txt | grep "Total execution time:" >> accumulated_time.txt
    cat accumulated_time.txt

run-schedules-gen-real-time-n-times:
    echo "1/3..."
    just run-schedules-gen-real-time-part-1
    just run-schedules-gen-real-time-part-2
    echo "2/3..."
    just run-schedules-gen-real-time-part-1
    just run-schedules-gen-real-time-part-2
    echo "3/3..."
    just run-schedules-gen-real-time-part-1
    just run-schedules-gen-real-time-part-2

    python3 parse_times.py --input accumulated_time.txt

cat-math:
    python3 scripts/gen_schedule/schedule.py