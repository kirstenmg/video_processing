import duckdb

con = duckdb.connect(database="benchmark.duckdb", read_only=False) 

print("do you want to drop tables? type 'yes' if so")
if input() == "yes":
	con.execute("DROP TABLE combo_full_benchmark");
	con.execute("DROP TABLE queue_block")
	con.execute("DROP TABLE experiment");

con.execute("CREATE TABLE experiment (eid BIGINT PRIMARY KEY, description VARCHAR)")
con.execute("CREATE TABLE combo_full_benchmark (eid BIGINT references experiment(eid), iteration INTEGER, pytorch_portion INTEGER, dali_portion INTEGER, clip_count INTEGER, clock_time FLOAT)")
con.execute("CREATE TABLE queue_block (eid BIGINT references experiment(eid), operation VARCHAR, pid INTEGER, dataloader VARCHAR, start_time TIMESTAMP, size_before INTEGER, size_after INTEGER, duration FLOAT)")
