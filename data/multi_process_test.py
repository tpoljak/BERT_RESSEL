import os
import time
from multiprocessing import Value, Process, Lock, Queue


def counter_test(val, lock, queue):
	for i in range(50):
		time.sleep(0.01)
		lock.acquire()
		val.value += 1
		queue.put(val.value)
		print(os.getpid(),":",val.value)
		lock.release()
	return val.value

def queue_test():
	while True:
		print("queue_get", queue.get())
		time.sleep(10)

queue = Queue()
curr_pos = Value('i', 0)
lock = Lock()
procs = [Process(target=counter_test, args=(curr_pos, lock, queue)) for i in range(2)]
proc_queue = Process(target=queue_test)

for proc in procs: proc.start()
proc_queue.start()

for proc in procs: proc.join()
proc_queue.join()