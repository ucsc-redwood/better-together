#!/usr/bin/env python3
import sqlite3
from sqlite3 import Error


def create_connection(db_file):
    """
    Create a database connection to an SQLite database.

    Parameters:
        db_file (str): The filename for the SQLite database.
    Returns:
        Connection object or None if connection cannot be created.
    """
    try:
        conn = sqlite3.connect(db_file)
        print(f"Connected to SQLite database: {db_file}")
        return conn
    except Error as e:
        print(f"Error creating connection: {e}")
        return None


def create_table(conn):
    """
    Create the benchmark_results table in the SQLite database.

    The table holds:
        - id: a unique row identifier.
        - stage: an integer (0 for baseline, 1-9 for other stages).
        - device: a text field describing the device ('CPU_little', 'CPU_mid', 'CPU_big', or 'GPU').
        - measurement: a real number representing the measured value.
        - timestamp: auto-generated timestamp for when the entry was inserted.
    """
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS benchmark_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        stage INTEGER NOT NULL,
        device TEXT NOT NULL,
        measurement REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
        conn.commit()
        print("Table 'benchmark_results' created or already exists.")
    except Error as e:
        print(f"Error creating table: {e}")


def insert_result(conn, stage, device, measurement):
    """
    Insert a new benchmark result into the benchmark_results table.

    Parameters:
        conn: Database connection object.
        stage (int): The stage number (0 for baseline, 1-9 for others).
        device (str): Identifier for the device mode (e.g., 'CPU_little', 'GPU').
        measurement (float): The measurement value for that stage/device.
    Returns:
        The id of the inserted row.
    """
    sql = """INSERT INTO benchmark_results(stage, device, measurement)
             VALUES (?, ?, ?)"""
    cur = conn.cursor()
    cur.execute(sql, (stage, device, measurement))
    conn.commit()
    return cur.lastrowid


def query_results(conn):
    """
    Fetch and display all records from the benchmark_results table.
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT id, stage, device, measurement, timestamp FROM benchmark_results"
    )
    rows = cur.fetchall()

    print("\nBenchmark Results:")
    for row in rows:
        print(row)


def main():
    database = "benchmark.db"

    # Create a database connection
    conn = create_connection(database)
    if conn is not None:
        # Create table if it doesn't exist
        create_table(conn)

        # Example: Insert a baseline (stage 0) record with an arbitrary measurement value.
        baseline_id = insert_result(conn, 0, "baseline", 1.23)
        print(f"Inserted baseline record with id: {baseline_id}")

        # Insert sample results for stage 1. You can replicate this for stages 1-9.
        devices = ["CPU_little", "CPU_mid", "CPU_big", "GPU"]
        stage = 1
        for device in devices:
            # Use a dummy measurement value (for example purposes)
            measurement = 2.0 + devices.index(device) * 1.1
            record_id = insert_result(conn, stage, device, measurement)
            print(f"Inserted record for stage {stage} on {device} with id: {record_id}")

        # Query and display all inserted records.
        query_results(conn)

        # Close the database connection.
        conn.close()
    else:
        print("Error! Cannot create the database connection.")


if __name__ == "__main__":
    main()
