# Collection Scripts


## `00_bm.py`

Run the benchmark on a single device single application on the single backend
Results will be stoed in the specified folder

```bash
uv run scripts/collect/00_bm.py --log_folder data/bm_logs --repeat 1 --app tree --backend vk --device 3A021JEHN02756
```

### `02_schedule.py` and `02_schedule_using_normal_table.py`

Using the specified folder as input (termed _profiling table_ in the paper), 
it will generated a JSON files of schedules.

```bash
uv run scripts/collect/02_schedule.py --csv_folder data/bm_logs/ --device 3A021JEHN02756 --app cifar-sparse --backend vk --num_solutions 30 --output_folder data/schedules/
```

