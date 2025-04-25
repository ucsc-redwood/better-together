baselines = {
    "3A021JEHN02756": {
        "cifar-dense-vk": {
            "omp": 940,
            "vk": 11.4,
        },
        "cifar-sparse-vk": {
            "omp": 45.8,
            "vk": 44.9,
        },
        "tree-vk": {
            "omp": 14.2,
            "vk": 58.7,
        },
    },
    "9b034f1b": {
        "cifar-dense-vk": {
            "omp": 730,
            "vk": 12.1,
        },
        "cifar-sparse-vk": {
            "omp": 53.2,
            "vk": 27.9,
        },
        "tree-vk": {
            "omp": 12.7,
            "vk": 47.2,
        },
    },
    "jetson": {
        "cifar-dense-cu": {
            "omp": 521,
            "cu": 27.2,
        },
        "cifar-sparse-cu": {
            "omp": 23.1,
            "cu": 5.48,
        },
        "tree-cu": {
            "omp": 16.6,
            "cu": 5.44,
        },
    },
    "jetsonlowpower": {
        "cifar-dense-cu": {
            "omp": 1067,
            "cu": 101,
        },
        "cifar-sparse-cu": {
            "omp": 60.7,
            "cu": 23.6,
        },
        "tree-cu": {
            "omp": 41.9,
            "cu": 7.26,
        },
    },
}


cifar_sparse_3a021jehn02756 = """
===== MEASURED VS PREDICTED TIMES =====
Schedule UID                    : Measured (ms)  Predicted (ms)  Difference (%)
--------------------------------------------------------------------------------
SCH-B1G3L1M4-G242-91e0         :         5.34            7.65          -30.18%
SCH-M1G3L1B4-G197-d983         :         5.38            7.86          -31.57%
SCH-M1G3B3L2-G221-aa43         :         4.23            7.86          -46.15%
SCH-B1G3M3L2-G264-55fd         :         3.96            7.86          -49.69%
SCH-G3M2B4-G208-03eb           :         7.67            9.95          -22.88%
SCH-G3B2M4-G277-491e           :         5.35            9.95          -46.18%
SCH-G3B1M5-G281-0d37           :         6.99            9.95          -29.77%
SCH-G3B1L1M4-G281-92d5         :         5.48            9.95          -44.91%
SCH-G3M1L1B4-G300-f8c0         :         5.86            9.95          -41.11%
SCH-G3M1B5-G300-924d           :         7.37            9.95          -25.95%
SCH-G4B5-G223-b4d8             :         8.38           11.95          -29.87%
SCH-G6L3-G046-d2bb             :        15.17           15.74           -3.61%
SCH-G9-G000-d386               :        33.44           19.39          +72.49%
SCH-B2M7-G111-f30e             :        15.01           19.48          -22.97%
SCH-M2B7-G067-d987             :        14.12           20.00          -29.43%
SCH-M4L5-G029-4af3             :        21.79           30.17          -27.79%
SCH-B4L5-G152-935b             :        22.17           30.17          -26.51%
SCH-B9-G000-c3e9               :        26.72           38.38          -30.38%
SCH-M9-G000-d2cb               :        30.19           38.81          -22.21%
SCH-L9-G000-7d07               :        68.61          108.77          -36.92%
"""

cifar_dense_3a021jehn02756 = """
===== MEASURED VS PREDICTED TIMES =====
Schedule UID                    : Measured (ms)  Predicted (ms)  Difference (%)
--------------------------------------------------------------------------------
SCH-G7L2-G195-0c02             :         6.20            6.54           -5.14%
SCH-G7B2-G499-110a             :         6.04            6.54           -7.53%
SCH-G7M2-G503-02ca             :         6.27            6.54           -4.00%
SCH-G7B1L1-G559-3f12           :         6.14            6.54           -6.03%
SCH-G7M1L1-G564-a5c2           :         5.49            6.54          -15.96%
SCH-G7L1M1-G593-3e59           :         5.57            6.54          -14.74%
SCH-G7B1M1-G593-5996           :         6.14            6.54           -6.06%
SCH-G7L1B1-G593-34a5           :         6.42            6.54           -1.79%
SCH-G7M1B1-G593-ba5e           :         6.00            6.54           -8.13%
SCH-G8L1-G553-ec03             :         6.63            6.78           -2.16%
SCH-G8M1-G617-1052             :         6.72            6.78           -0.83%
SCH-G8B1-G617-1601             :         6.56            6.78           -3.19%
SCH-G9-G000-de67               :         9.55            7.54          +26.59%
SCH-B1G8-G5149-0ef5            :         7.27           58.31          -87.52%
SCH-B1L1G7-G5184-812c          :         6.56           58.31          -88.76%
SCH-B1G6L2-G5372-1b82          :         5.80           58.31          -90.05%
SCH-M6L3-G1769-caf2            :       538.67          718.34          -25.01%
SCH-M9-G000-027e               :       793.40         1067.89          -25.70%
SCH-B9-G000-0f7f               :       895.68         1136.26          -21.17%
SCH-L9-G000-416a               :      2008.69         2388.01          -15.88%
"""

tree_3a021jehn02756 = """
===== MEASURED VS PREDICTED TIMES =====
Schedule UID                    : Measured (ms)  Predicted (ms)  Difference (%)
--------------------------------------------------------------------------------
SCH-B2G2L2M1-G042-fedd         :         2.05            2.41          -15.16%
SCH-M2G2L2B1-G045-683f         :         1.69            2.54          -33.53%
SCH-L1M2G3B1-G054-957a         :         0.76            2.62          -71.11%
SCH-L1B2G3M1-G065-ff79         :         0.87            2.62          -66.94%
SCH-L1M2G2B2-G074-dae0         :         1.52            2.62          -41.96%
SCH-L1B2G2M2-G074-b58e         :         1.81            2.62          -30.77%
SCH-B3G3M1-G066-7162           :         1.38            2.66          -48.05%
SCH-M2G2B3-G038-90df           :         2.33            2.80          -16.56%
SCH-M2L1G3B1-G077-1eb3         :         1.07            2.85          -62.55%
SCH-B2G2M3-G054-076e           :         2.64            2.86           -7.58%
SCH-M2G3B2-G045-70c0           :         2.41            2.98          -19.23%
SCH-B2G3M2-G066-fe33           :         3.33            2.98          +11.47%
SCH-B4L3-G029-6a78             :         5.26            8.24          -36.25%
SCH-M4L3-G004-dff0             :         6.33            8.29          -23.67%
SCH-L3M4-G058-b1fa             :         3.80            8.82          -56.88%
SCH-L3B4-G073-d1c7             :         3.63            8.82          -58.82%
SCH-B7-G000-5c06               :         6.92           10.75          -35.65%
SCH-M7-G000-02e4               :         8.34           11.14          -25.12%
SCH-L7-G000-ad72               :        18.06           26.52          -31.91%
SCH-G7-G000-d1bd               :        55.01           27.45         +100.38%
"""

tree_9b034f1b = """
===== MEASURED VS PREDICTED TIMES =====
Schedule UID                    : Measured (ms)  Predicted (ms)  Difference (%)
--------------------------------------------------------------------------------
SCH-B2G3M2-G059-4530           :         2.84            2.76           +2.83%
SCH-B2G2M3-G082-0a3d           :         2.12            2.76          -23.24%
SCH-B2G4M1-G088-ffd0           :         3.78            2.76          +37.07%
SCH-B2G2L2M1-G113-cc48         :         1.54            2.76          -44.23%
SCH-M2G2B3-G091-9857           :         2.05            2.85          -28.13%
SCH-M2G3B2-G109-1546           :         2.81            2.85           -1.38%
SCH-B3G2M2-G131-5ea7           :         1.89            3.26          -42.21%
SCH-M3G4-G028-4606             :         3.62            3.56           +1.75%
SCH-B3G4-G030-86ec             :         3.41            3.56           -4.11%
SCH-L1B2G4-G109-0669           :         2.12            3.72          -42.94%
SCH-M2G5-G112-99b4             :         5.82            3.97          +46.65%
SCH-B2G5-G121-df6b             :         5.52            3.97          +39.02%
SCH-L2B5-G023-807f             :         5.91            6.83          -13.40%
SCH-B5L2-G102-3278             :         6.06            7.60          -20.31%
SCH-B4L3-G039-babf             :         6.01            7.81          -23.11%
SCH-L3M4-G113-9007             :         2.52            8.68          -70.93%
SCH-B7-G000-ce4f               :        13.13            9.36          +40.28%
SCH-M7-G000-187e               :        10.79           11.97           -9.80%
SCH-L7-G000-9059               :        15.34           24.53          -37.44%
SCH-G7-G000-99a5               :        46.94           36.69          +27.93%
"""

cifar_dense_9b034f1b = """
===== MEASURED VS PREDICTED TIMES =====
Schedule UID                    : Measured (ms)  Predicted (ms)  Difference (%)
--------------------------------------------------------------------------------
SCH-G7M2-G323-e255             :         7.77            4.95          +56.78%
SCH-G7B2-G386-d64d             :         6.41            4.95          +29.34%
SCH-G7M1L1-G397-3897           :         3.99            4.95          -19.42%
SCH-G7L1M1-G422-3eee           :         3.86            4.95          -22.07%
SCH-G7B1M1-G432-cae1           :         6.62            4.95          +33.55%
SCH-G7B1L1-G432-fdb9           :         3.57            4.95          -27.99%
SCH-G7L1B1-G449-8fa9           :         3.59            4.95          -27.59%
SCH-G7M1B1-G449-cb75           :         6.65            4.95          +34.26%
SCH-G7L2-G011-3949             :         4.73            5.06           -6.60%
SCH-G8L1-G402-6c89             :         5.38            5.23           +2.86%
SCH-G8M1-G449-e234             :         8.83            5.23          +68.74%
SCH-G8B1-G477-c491             :         7.42            5.23          +41.90%
SCH-G9-G000-3a57               :        11.03            6.68          +65.07%
SCH-B1G8-G4117-7f71            :         8.57           46.97          -81.76%
SCH-B1L1G7-G4155-ce86          :         4.95           46.97          -89.45%
SCH-B1G6L2-G4290-e19e          :         3.18           46.97          -93.24%
SCH-B1M1G7-G4301-f366          :         7.46           46.97          -84.11%
SCH-B9-G000-d9d4               :       736.38          776.98           -5.22%
SCH-M9-G000-3525               :      1153.79         1276.56           -9.62%
SCH-L9-G000-3cee               :      1646.59         2153.07          -23.52%
"""

cifar_sparse_9b034f1b = """
===== MEASURED VS PREDICTED TIMES =====
Schedule UID                    : Measured (ms)  Predicted (ms)  Difference (%)
--------------------------------------------------------------------------------
SCH-M1G3B5-G064-f4c0           :         5.69            7.05          -19.22%
SCH-M1G4B4-G258-4358           :         7.07            8.19          -13.67%
SCH-B1G4M4-G348-491e           :         6.96            8.19          -15.03%
SCH-G3M1B5-G170-59f8           :         6.64            8.75          -24.06%
SCH-G3B2M4-G220-8d2f           :         6.54            8.75          -25.26%
SCH-G3B3M3-G258-1fb4           :         6.27            8.75          -28.34%
SCH-G2B2M3L2-G284-12e0         :         5.08            8.77          -42.06%
SCH-G2B3M4-G274-72cc           :         6.57            9.04          -27.32%
SCH-G3B4L2-G065-9d53           :         4.91            9.40          -47.80%
SCH-M1B1G7-G349-775e           :        10.48            9.90           +5.91%
SCH-G4M5-G014-0a92             :         9.12           10.28          -11.20%
SCH-G4B5-G323-7ee9             :        10.79           10.28           +4.97%
SCH-G4L1M4-G242-d6ad           :         6.82           10.58          -35.53%
SCH-G3B6-G342-62cc             :        13.21           12.16           +8.59%
SCH-G9-G000-923b               :        22.39           15.59          +43.61%
SCH-B3M6-G093-ebfd             :        12.36           18.04          -31.49%
SCH-B9-G000-ba53               :        44.39           29.27          +51.67%
SCH-L1M8-G088-4836             :        23.16           36.77          -37.01%
SCH-M9-G000-c96f               :        39.48           43.18           -8.56%
SCH-L9-G000-68ca               :        93.52          139.74          -33.07%
"""

jetson_cifar_dense_cu = """
Schedule UID                    : Measured (ms)  Predicted (ms)  Difference (%)
--------------------------------------------------------------------------------
SCH-G7L2-G2833-89f7            :        26.02           30.50          -14.70%
SCH-G8L1-G3019-5eb3            :        26.00           30.82          -15.62%
SCH-G9-G000-aae9               :        26.18           31.34          -16.47%
SCH-L1G8-G822-946f             :        26.04           37.70          -30.94%
SCH-L2G7-G1352-c501            :        22.34           42.49          -47.41%
SCH-L3G6-G10765-ff94           :       126.16          132.34           -4.67%
SCH-L4G5-G11117-4efd           :       132.10          135.51           -2.52%
SCH-G6L3-G15376-fb7e           :        17.52          174.92          -89.98%
SCH-L5G4-G20303-8140           :       286.71          222.61          +28.79%
SCH-G5L4-G33545-d24b           :       231.05          347.21          -33.46%
SCH-L6G3-G38472-f17c           :       565.80          394.90          +43.28%
SCH-G4L5-G42731-1f70           :       471.84          434.31           +8.64%
SCH-G3L6-G43083-acb5           :       596.61          437.49          +36.37%
SCH-G2L7-G52496-6c68           :       603.00          527.33          +14.35%
SCH-G1L8-G53026-799b           :       740.92          532.12          +39.24%
SCH-L7G2-G56681-6f78           :       836.39          567.65          +47.34%
SCH-L8G1-G56867-ac0e           :       840.40          569.20          +47.65%
SCH-L9-G000-9c0e               :       526.50          569.82           -7.60%
"""

jetson_cifar_sparse_cu = """
===== MEASURED VS PREDICTED TIMES =====
Schedule UID                    : Measured (ms)  Predicted (ms)  Difference (%)
--------------------------------------------------------------------------------
SCH-G4L5-G174-0076             :         5.69            9.24          -38.37%
SCH-G5L4-G526-6ca1             :         4.80           11.30          -57.51%
SCH-G3L6-G440-ff76             :         7.62           12.41          -38.58%
SCH-L1G8-G835-818a             :         4.03           12.67          -68.19%
SCH-G6L3-G869-7c53             :         4.21           13.24          -68.24%
SCH-L2G7-G300-797a             :        13.14           14.23           -7.68%
SCH-G2L7-G966-22ae             :        14.82           14.87           -0.32%
SCH-G7L2-G1199-47a1            :         4.43           15.04          -70.54%
SCH-G8L1-G1595-1e29            :         4.61           15.99          -71.19%
SCH-G9-G000-bf91               :         4.84           16.44          -70.53%
SCH-L3G6-G826-b414             :        16.72           16.69           +0.18%
SCH-L4G5-G1441-1bc8            :        23.60           21.61           +9.23%
SCH-L5G4-G1793-90cf            :        26.02           23.06          +12.84%
SCH-L6G3-G2136-4414            :        28.14           24.55          +14.62%
SCH-G1L8-G2102-fcc0            :        18.54           24.78          -25.17%
SCH-L7G2-G2465-c39c            :        30.29           26.05          +16.24%
SCH-L8G1-G2861-b312            :        34.85           29.06          +19.91%
SCH-L9-G000-7410               :        20.04           29.10          -31.15%
"""

jetson_tree_cu = """
===== MEASURED VS PREDICTED TIMES =====
Schedule UID                    : Measured (ms)  Predicted (ms)  Difference (%)
--------------------------------------------------------------------------------
SCH-G4L3-G289-1a0c             :         5.45            5.59           -2.39%
SCH-G5L2-G350-aea7             :         5.04            5.88          -14.26%
SCH-L2G5-G010-88c3             :         4.67            6.51          -28.26%
SCH-G3L4-G255-ef24             :         4.64            7.07          -34.46%
SCH-L3G4-G305-e03e             :         5.65            7.43          -24.00%
SCH-G6L1-G610-beab             :         6.89            7.64           -9.78%
SCH-G2L5-G550-e2d3             :         8.72            7.99           +9.17%
SCH-L1G6-G742-7f33             :         5.23            8.55          -38.82%
SCH-G7-G000-e0b6               :         6.87            8.90          -22.84%
SCH-L4G3-G849-2e19             :        12.37           11.81           +4.73%
SCH-L5G2-G909-85ad             :        12.28           12.11           +1.34%
SCH-L6G1-G1170-e3fc            :        13.34           12.97           +2.87%
SCH-G1L6-G1302-7954            :         9.19           13.37          -31.27%
SCH-L7-G000-a614               :         9.63           14.50          -33.60%
"""

jetsonlowpower_cifar_dense_cu = """
===== MEASURED VS PREDICTED TIMES =====
Schedule UID                    : Measured (ms)  Predicted (ms)  Difference (%)
--------------------------------------------------------------------------------
SCH-L2G7-G1999-2167            :        78.37          107.95          -27.41%
SCH-L1G8-G2978-030e            :        93.90          109.00          -13.86%
SCH-G7L2-G11106-88da           :        93.82          115.59          -18.83%
SCH-G8L1-G11491-6228           :        93.90          116.03          -19.07%
SCH-G9-G000-471d               :        94.51          117.02          -19.24%
SCH-L3G6-G18806-c282           :       271.79          277.74           -2.14%
SCH-L4G5-G19400-72bd           :       289.17          283.08           +2.15%
SCH-G6L3-G30160-0ed5           :        63.22          381.82          -83.44%
SCH-L5G4-G39630-c4bc           :       605.73          467.70          +29.51%
SCH-G5L4-G70514-1061           :       473.95          750.77          -36.87%
SCH-L6G3-G79984-6b42           :      1155.38          836.65          +38.10%
SCH-G4L5-G90744-8a44           :       974.78          935.39           +4.21%
SCH-G3L6-G91338-ad7c           :      1241.44          940.72          +31.97%
SCH-G2L7-G112143-5564          :      1257.38         1130.50          +11.22%
SCH-G1L8-G113122-bafa          :      1529.17         1139.25          +34.23%
SCH-L7G2-G121250-80b6          :      1688.36         1213.93          +39.08%
SCH-L8G1-G121635-0ae0          :      1699.73         1217.34          +39.63%
SCH-L9-G000-e6d3               :      1184.80         1218.46           -2.76%
"""

jetsonlowpower_cifar_sparse_cu = """
===== MEASURED VS PREDICTED TIMES =====
Schedule UID                    : Measured (ms)  Predicted (ms)  Difference (%)
--------------------------------------------------------------------------------
SCH-G4L5-G548-892d             :        10.45           20.80          -49.77%
SCH-L1G8-G1441-530e            :        13.24           23.26          -43.06%
SCH-G5L4-G1221-eea4            :        11.33           24.49          -53.73%
SCH-G3L6-G651-79dc             :        15.55           25.43          -38.85%
SCH-L2G7-G670-36cf             :        22.23           26.75          -16.92%
SCH-G6L3-G1878-b658            :        13.17           28.03          -53.00%
SCH-G2L7-G1712-db22            :        31.16           30.34           +2.69%
SCH-G7L2-G2521-db1d            :        15.09           31.44          -52.02%
SCH-L3G6-G1731-258b            :        31.57           31.66           -0.29%
SCH-G8L1-G3273-0b60            :        15.87           32.77          -51.57%
SCH-G9-G000-2d80               :        15.79           33.28          -52.54%
SCH-L4G5-G2930-b959            :        48.75           41.77          +16.69%
SCH-L5G4-G3603-fa79            :        55.07           44.82          +22.87%
SCH-L6G3-G4260-9ca9            :        66.15           47.84          +38.25%
SCH-G1L8-G3822-be5e            :        43.42           48.25          -10.01%
SCH-L7G2-G4903-cb43            :        69.96           50.86          +37.55%
SCH-L8G1-G5655-90a6            :        80.38           57.05          +40.90%
SCH-L9-G000-b066               :        44.85           57.10          -21.45%
"""

jetsonlowpower_tree_cu = """
===== MEASURED VS PREDICTED TIMES =====
Schedule UID                    : Measured (ms)  Predicted (ms)  Difference (%)
--------------------------------------------------------------------------------
SCH-G4L3-G551-a4ea             :         9.87           10.18           -3.09%
SCH-L3G4-G174-4a84             :         9.83           10.43           -5.81%
SCH-G5L2-G673-4012             :         8.95           10.82          -17.24%
SCH-L2G5-G287-4624             :         7.06           11.97          -41.03%
SCH-G3L4-G515-2e6d             :         7.59           12.87          -41.02%
SCH-G2L5-G975-5b1a             :        18.23           14.20          +28.37%
SCH-G6L1-G1136-2b9c            :        10.33           14.21          -27.34%
SCH-L1G6-G1370-a2bc            :         9.48           15.82          -40.07%
SCH-G7-G000-f962               :        11.14           16.42          -32.15%
SCH-L4G3-G1240-f2a2            :        23.29           18.63          +25.02%
SCH-L5G2-G1361-7cfd            :        22.87           19.21          +19.03%
SCH-L6G1-G1824-2f6a            :        26.40           20.45          +29.09%
SCH-G1L6-G2058-c743            :        25.85           21.18          +22.02%
SCH-L7-G000-02f2               :        26.59           23.30          +14.12%
"""

import pandas as pd
import io
import numpy as np
from scipy import stats


def parse_speedup_data(data_string):
    # Skip the header lines and parse the data
    lines = data_string.strip().split("\n")[4:]  # Skip the first 4 lines

    # Create lists to store the data
    schedules = []
    measured_times = []
    predicted_times = []
    differences = []

    for line in lines:
        parts = line.split(":")
        if len(parts) < 2:
            continue

        schedule = parts[0].strip()
        time_parts = parts[1].strip().split()

        if len(time_parts) >= 3:
            measured = float(time_parts[0])
            predicted = float(time_parts[1])
            difference = time_parts[2]

            schedules.append(schedule)
            measured_times.append(measured)
            predicted_times.append(predicted)
            differences.append(difference)

    # Create a DataFrame
    df = pd.DataFrame(
        {
            "Schedule": schedules,
            "Measured (ms)": measured_times,
            "Predicted (ms)": predicted_times,
            "Difference (%)": differences,
        }
    )

    return df


# Parse each dataset
cifar_sparse_3a021jehn02756_df = parse_speedup_data(cifar_sparse_3a021jehn02756)
cifar_sparse_9b034f1b_df = parse_speedup_data(cifar_sparse_9b034f1b)
cifar_dense_3a021jehn02756_df = parse_speedup_data(cifar_dense_3a021jehn02756)
cifar_dense_9b034f1b_df = parse_speedup_data(cifar_dense_9b034f1b)
tree_3a021jehn02756_df = parse_speedup_data(tree_3a021jehn02756)
tree_9b034f1b_df = parse_speedup_data(tree_9b034f1b)

jetson_cifar_dense_cu_df = parse_speedup_data(jetson_cifar_dense_cu)
jetson_cifar_sparse_cu_df = parse_speedup_data(jetson_cifar_sparse_cu)
jetson_tree_cu_df = parse_speedup_data(jetson_tree_cu)
jetsonlowpower_cifar_dense_cu_df = parse_speedup_data(jetsonlowpower_cifar_dense_cu)
jetsonlowpower_cifar_sparse_cu_df = parse_speedup_data(jetsonlowpower_cifar_sparse_cu)
jetsonlowpower_tree_cu_df = parse_speedup_data(jetsonlowpower_tree_cu)

# Get baseline times for each device and application
baseline_times = {
    "3A021JEHN02756": {
        "cifar-dense": {
            "cpu": baselines["3A021JEHN02756"]["cifar-dense-vk"]["omp"],
            "gpu": baselines["3A021JEHN02756"]["cifar-dense-vk"]["vk"],
        },
        "cifar-sparse": {
            "cpu": baselines["3A021JEHN02756"]["cifar-sparse-vk"]["omp"],
            "gpu": baselines["3A021JEHN02756"]["cifar-sparse-vk"]["vk"],
        },
        "tree": {
            "cpu": baselines["3A021JEHN02756"]["tree-vk"]["omp"],
            "gpu": baselines["3A021JEHN02756"]["tree-vk"]["vk"],
        },
    },
    "9b034f1b": {
        "cifar-dense": {
            "cpu": baselines["9b034f1b"]["cifar-dense-vk"]["omp"],
            "gpu": baselines["9b034f1b"]["cifar-dense-vk"]["vk"],
        },
        "cifar-sparse": {
            "cpu": baselines["9b034f1b"]["cifar-sparse-vk"]["omp"],
            "gpu": baselines["9b034f1b"]["cifar-sparse-vk"]["vk"],
        },
        "tree": {
            "cpu": baselines["9b034f1b"]["tree-vk"]["omp"],
            "gpu": baselines["9b034f1b"]["tree-vk"]["vk"],
        },
    },
    "jetson": {
        "cifar-dense": {
            "cpu": baselines["jetson"]["cifar-dense-cu"]["omp"],
            "gpu": baselines["jetson"]["cifar-dense-cu"]["cu"],
        },
        "cifar-sparse": {
            "cpu": baselines["jetson"]["cifar-sparse-cu"]["omp"],
            "gpu": baselines["jetson"]["cifar-sparse-cu"]["cu"],
        },
        "tree": {
            "cpu": baselines["jetson"]["tree-cu"]["omp"],
            "gpu": baselines["jetson"]["tree-cu"]["cu"],
        },
    },
    "jetsonlowpower": {
        "cifar-dense": {
            "cpu": baselines["jetsonlowpower"]["cifar-dense-cu"]["omp"],
            "gpu": baselines["jetsonlowpower"]["cifar-dense-cu"]["cu"],
        },
        "cifar-sparse": {
            "cpu": baselines["jetsonlowpower"]["cifar-sparse-cu"]["omp"],
            "gpu": baselines["jetsonlowpower"]["cifar-sparse-cu"]["cu"],
        },
        "tree": {
            "cpu": baselines["jetsonlowpower"]["tree-cu"]["omp"],
            "gpu": baselines["jetsonlowpower"]["tree-cu"]["cu"],
        },
    },
}


# Function to get best predicted time from a dataframe
def get_best_predicted_time(df):
    return df["Predicted (ms)"].min()


# Function to get the faster baseline time
def get_faster_baseline(device, app):
    cpu_time = baseline_times[device][app]["cpu"]
    gpu_time = baseline_times[device][app]["gpu"]
    return min(cpu_time, gpu_time)


# Compute speedups for each application and device against both CPU and GPU baselines
speedups = {
    "3A021JEHN02756": {
        "cifar-dense": {
            "cpu": baseline_times["3A021JEHN02756"]["cifar-dense"]["cpu"]
            / get_best_predicted_time(cifar_dense_3a021jehn02756_df),
            "gpu": baseline_times["3A021JEHN02756"]["cifar-dense"]["gpu"]
            / get_best_predicted_time(cifar_dense_3a021jehn02756_df),
        },
        "cifar-sparse": {
            "cpu": baseline_times["3A021JEHN02756"]["cifar-sparse"]["cpu"]
            / get_best_predicted_time(cifar_sparse_3a021jehn02756_df),
            "gpu": baseline_times["3A021JEHN02756"]["cifar-sparse"]["gpu"]
            / get_best_predicted_time(cifar_sparse_3a021jehn02756_df),
        },
        "tree": {
            "cpu": baseline_times["3A021JEHN02756"]["tree"]["cpu"]
            / get_best_predicted_time(tree_3a021jehn02756_df),
            "gpu": baseline_times["3A021JEHN02756"]["tree"]["gpu"]
            / get_best_predicted_time(tree_3a021jehn02756_df),
        },
    },
    "9b034f1b": {
        "cifar-dense": {
            "cpu": baseline_times["9b034f1b"]["cifar-dense"]["cpu"]
            / get_best_predicted_time(cifar_dense_9b034f1b_df),
            "gpu": baseline_times["9b034f1b"]["cifar-dense"]["gpu"]
            / get_best_predicted_time(cifar_dense_9b034f1b_df),
        },
        "cifar-sparse": {
            "cpu": baseline_times["9b034f1b"]["cifar-sparse"]["cpu"]
            / get_best_predicted_time(cifar_sparse_9b034f1b_df),
            "gpu": baseline_times["9b034f1b"]["cifar-sparse"]["gpu"]
            / get_best_predicted_time(cifar_sparse_9b034f1b_df),
        },
        "tree": {
            "cpu": baseline_times["9b034f1b"]["tree"]["cpu"]
            / get_best_predicted_time(tree_9b034f1b_df),
            "gpu": baseline_times["9b034f1b"]["tree"]["gpu"]
            / get_best_predicted_time(tree_9b034f1b_df),
        },
    },
    "jetson": {
        "cifar-dense": {
            "cpu": baseline_times["jetson"]["cifar-dense"]["cpu"]
            / get_best_predicted_time(jetson_cifar_dense_cu_df),
            "gpu": baseline_times["jetson"]["cifar-dense"]["gpu"]
            / get_best_predicted_time(jetson_cifar_dense_cu_df),
        },
        "cifar-sparse": {
            "cpu": baseline_times["jetson"]["cifar-sparse"]["cpu"]
            / get_best_predicted_time(jetson_cifar_sparse_cu_df),
            "gpu": baseline_times["jetson"]["cifar-sparse"]["gpu"]
            / get_best_predicted_time(jetson_cifar_sparse_cu_df),
        },
        "tree": {
            "cpu": baseline_times["jetson"]["tree"]["cpu"]
            / get_best_predicted_time(jetson_tree_cu_df),
            "gpu": baseline_times["jetson"]["tree"]["gpu"]
            / get_best_predicted_time(jetson_tree_cu_df),
        },
    },
    "jetsonlowpower": {
        "cifar-dense": {
            "cpu": baseline_times["jetsonlowpower"]["cifar-dense"]["cpu"]
            / get_best_predicted_time(jetsonlowpower_cifar_dense_cu_df),
            "gpu": baseline_times["jetsonlowpower"]["cifar-dense"]["gpu"]
            / get_best_predicted_time(jetsonlowpower_cifar_dense_cu_df),
        },
        "cifar-sparse": {
            "cpu": baseline_times["jetsonlowpower"]["cifar-sparse"]["cpu"]
            / get_best_predicted_time(jetsonlowpower_cifar_sparse_cu_df),
            "gpu": baseline_times["jetsonlowpower"]["cifar-sparse"]["gpu"]
            / get_best_predicted_time(jetsonlowpower_cifar_sparse_cu_df),
        },
        "tree": {
            "cpu": baseline_times["jetsonlowpower"]["tree"]["cpu"]
            / get_best_predicted_time(jetsonlowpower_tree_cu_df),
            "gpu": baseline_times["jetsonlowpower"]["tree"]["gpu"]
            / get_best_predicted_time(jetsonlowpower_tree_cu_df),
        },
    },
}

# Print individual speedups
print("\nSpeedups for each application and device:")
for device in speedups:
    print(f"\n{device}:")
    for app in speedups[device]:
        print(f"  {app}:")
        print(f"    CPU baseline: {speedups[device][app]['cpu']:.2f}x")
        print(f"    GPU baseline: {speedups[device][app]['gpu']:.2f}x")

# Calculate separate geomeans for CPU and GPU baselines
cpu_speedups = [
    speedups[device][app]["cpu"] for device in speedups for app in speedups[device]
]
gpu_speedups = [
    speedups[device][app]["gpu"] for device in speedups for app in speedups[device]
]

cpu_geomean = np.exp(np.mean(np.log(cpu_speedups)))
gpu_geomean = np.exp(np.mean(np.log(gpu_speedups)))

print(f"\nGeometric mean of CPU baseline speedups: {cpu_geomean:.2f}x")
print(f"Geometric mean of GPU baseline speedups: {gpu_geomean:.2f}x")


# Function to get measured time of the fastest predicted schedule
def get_measured_time_of_fastest_predicted(df):
    fastest_predicted_idx = df["Predicted (ms)"].idxmin()
    return df.loc[fastest_predicted_idx, "Measured (ms)"]


# Compute speedups using measured times of fastest predicted schedules
measured_speedups = {
    "3A021JEHN02756": {
        "cifar-dense": {
            "cpu": baseline_times["3A021JEHN02756"]["cifar-dense"]["cpu"]
            / get_measured_time_of_fastest_predicted(cifar_dense_3a021jehn02756_df),
            "gpu": baseline_times["3A021JEHN02756"]["cifar-dense"]["gpu"]
            / get_measured_time_of_fastest_predicted(cifar_dense_3a021jehn02756_df),
        },
        "cifar-sparse": {
            "cpu": baseline_times["3A021JEHN02756"]["cifar-sparse"]["cpu"]
            / get_measured_time_of_fastest_predicted(cifar_sparse_3a021jehn02756_df),
            "gpu": baseline_times["3A021JEHN02756"]["cifar-sparse"]["gpu"]
            / get_measured_time_of_fastest_predicted(cifar_sparse_3a021jehn02756_df),
        },
        "tree": {
            "cpu": baseline_times["3A021JEHN02756"]["tree"]["cpu"]
            / get_measured_time_of_fastest_predicted(tree_3a021jehn02756_df),
            "gpu": baseline_times["3A021JEHN02756"]["tree"]["gpu"]
            / get_measured_time_of_fastest_predicted(tree_3a021jehn02756_df),
        },
    },
    "9b034f1b": {
        "cifar-dense": {
            "cpu": baseline_times["9b034f1b"]["cifar-dense"]["cpu"]
            / get_measured_time_of_fastest_predicted(cifar_dense_9b034f1b_df),
            "gpu": baseline_times["9b034f1b"]["cifar-dense"]["gpu"]
            / get_measured_time_of_fastest_predicted(cifar_dense_9b034f1b_df),
        },
        "cifar-sparse": {
            "cpu": baseline_times["9b034f1b"]["cifar-sparse"]["cpu"]
            / get_measured_time_of_fastest_predicted(cifar_sparse_9b034f1b_df),
            "gpu": baseline_times["9b034f1b"]["cifar-sparse"]["gpu"]
            / get_measured_time_of_fastest_predicted(cifar_sparse_9b034f1b_df),
        },
        "tree": {
            "cpu": baseline_times["9b034f1b"]["tree"]["cpu"]
            / get_measured_time_of_fastest_predicted(tree_9b034f1b_df),
            "gpu": baseline_times["9b034f1b"]["tree"]["gpu"]
            / get_measured_time_of_fastest_predicted(tree_9b034f1b_df),
        },
    },
    "jetson": {
        "cifar-dense": {
            "cpu": baseline_times["jetson"]["cifar-dense"]["cpu"]
            / get_measured_time_of_fastest_predicted(jetson_cifar_dense_cu_df),
            "gpu": baseline_times["jetson"]["cifar-dense"]["gpu"]
            / get_measured_time_of_fastest_predicted(jetson_cifar_dense_cu_df),
        },
        "cifar-sparse": {
            "cpu": baseline_times["jetson"]["cifar-sparse"]["cpu"]
            / get_measured_time_of_fastest_predicted(jetson_cifar_sparse_cu_df),
            "gpu": baseline_times["jetson"]["cifar-sparse"]["gpu"]
            / get_measured_time_of_fastest_predicted(jetson_cifar_sparse_cu_df),
        },
        "tree": {
            "cpu": baseline_times["jetson"]["tree"]["cpu"]
            / get_measured_time_of_fastest_predicted(jetson_tree_cu_df),
            "gpu": baseline_times["jetson"]["tree"]["gpu"]
            / get_measured_time_of_fastest_predicted(jetson_tree_cu_df),
        },
    },
    "jetsonlowpower": {
        "cifar-dense": {
            "cpu": baseline_times["jetsonlowpower"]["cifar-dense"]["cpu"]
            / get_measured_time_of_fastest_predicted(jetsonlowpower_cifar_dense_cu_df),
            "gpu": baseline_times["jetsonlowpower"]["cifar-dense"]["gpu"]
            / get_measured_time_of_fastest_predicted(jetsonlowpower_cifar_dense_cu_df),
        },
        "cifar-sparse": {
            "cpu": baseline_times["jetsonlowpower"]["cifar-sparse"]["cpu"]
            / get_measured_time_of_fastest_predicted(jetsonlowpower_cifar_sparse_cu_df),
            "gpu": baseline_times["jetsonlowpower"]["cifar-sparse"]["gpu"]
            / get_measured_time_of_fastest_predicted(jetsonlowpower_cifar_sparse_cu_df),
        },
        "tree": {
            "cpu": baseline_times["jetsonlowpower"]["tree"]["cpu"]
            / get_measured_time_of_fastest_predicted(jetsonlowpower_tree_cu_df),
            "gpu": baseline_times["jetsonlowpower"]["tree"]["gpu"]
            / get_measured_time_of_fastest_predicted(jetsonlowpower_tree_cu_df),
        },
    },
}

print("\nSpeedups using predicted times:")
for device in speedups:
    print(f"\n{device}:")
    for app in speedups[device]:
        print(f"  {app}:")
        print(f"    CPU baseline: {speedups[device][app]['cpu']:.2f}x")
        print(f"    GPU baseline: {speedups[device][app]['gpu']:.2f}x")

print("\nSpeedups using measured times of fastest predicted schedules:")
for device in measured_speedups:
    print(f"\n{device}:")
    for app in measured_speedups[device]:
        print(f"  {app}:")
        print(f"    CPU baseline: {measured_speedups[device][app]['cpu']:.2f}x")
        print(f"    GPU baseline: {measured_speedups[device][app]['gpu']:.2f}x")

# Calculate separate geomeans for CPU and GPU baselines (predicted)
cpu_speedups = [
    speedups[device][app]["cpu"] for device in speedups for app in speedups[device]
]
gpu_speedups = [
    speedups[device][app]["gpu"] for device in speedups for app in speedups[device]
]

cpu_geomean = np.exp(np.mean(np.log(cpu_speedups)))
gpu_geomean = np.exp(np.mean(np.log(gpu_speedups)))

print(f"\nGeometric mean of CPU baseline speedups (predicted): {cpu_geomean:.2f}x")
print(f"Geometric mean of GPU baseline speedups (predicted): {gpu_geomean:.2f}x")

# Calculate separate geomeans for CPU and GPU baselines (measured)
measured_cpu_speedups = [
    measured_speedups[device][app]["cpu"]
    for device in measured_speedups
    for app in measured_speedups[device]
]
measured_gpu_speedups = [
    measured_speedups[device][app]["gpu"]
    for device in measured_speedups
    for app in measured_speedups[device]
]

measured_cpu_geomean = np.exp(np.mean(np.log(measured_cpu_speedups)))
measured_gpu_geomean = np.exp(np.mean(np.log(measured_gpu_speedups)))

print(
    f"\nGeometric mean of CPU baseline speedups (measured): {measured_cpu_geomean:.2f}x"
)
print(
    f"Geometric mean of GPU baseline speedups (measured): {measured_gpu_geomean:.2f}x"
)

# Find and print maximum speedups for both predicted and measured times
max_cpu_speedup_predicted = max(cpu_speedups)
max_gpu_speedup_predicted = max(gpu_speedups)
max_cpu_speedup_measured = max(measured_cpu_speedups)
max_gpu_speedup_measured = max(measured_gpu_speedups)

print("\n=== Maximum Speedups ===")
print(f"Maximum CPU baseline speedup (predicted): {max_cpu_speedup_predicted:.2f}x")
print(f"Maximum GPU baseline speedup (predicted): {max_gpu_speedup_predicted:.2f}x")
print(f"Maximum CPU baseline speedup (measured): {max_cpu_speedup_measured:.2f}x")
print(f"Maximum GPU baseline speedup (measured): {max_gpu_speedup_measured:.2f}x")


# Print which app/device combination achieved the maximum speedup
def find_max_speedup_info(speedups_dict, baseline_type):
    max_speedup = 0
    max_device = None
    max_app = None

    for device in speedups_dict:
        for app in speedups_dict[device]:
            if speedups_dict[device][app][baseline_type] > max_speedup:
                max_speedup = speedups_dict[device][app][baseline_type]
                max_device = device
                max_app = app

    return max_device, max_app, max_speedup


# Find which combinations achieved maximum speedups
max_cpu_device_pred, max_cpu_app_pred, _ = find_max_speedup_info(speedups, "cpu")
max_gpu_device_pred, max_gpu_app_pred, _ = find_max_speedup_info(speedups, "gpu")
max_cpu_device_meas, max_cpu_app_meas, _ = find_max_speedup_info(
    measured_speedups, "cpu"
)
max_gpu_device_meas, max_gpu_app_meas, _ = find_max_speedup_info(
    measured_speedups, "gpu"
)

print("\n=== Maximum Speedup Details ===")
print(f"Max CPU speedup (predicted): {max_cpu_device_pred}, {max_cpu_app_pred}")
print(f"Max GPU speedup (predicted): {max_gpu_device_pred}, {max_gpu_app_pred}")
print(f"Max CPU speedup (measured): {max_cpu_device_meas}, {max_cpu_app_meas}")
print(f"Max GPU speedup (measured): {max_gpu_device_meas}, {max_gpu_app_meas}")
