from Queue import Queue
import sys

convx = [2,3,4,5]
convy = [2,3,4,5]

start = (14, 34)
goal = (129, 285)

path = {
	start: []
}

q = Queue()
q.put(start)

while not q.empty():
	cur = q.get()
	for x in convx:
		for y in convy:
			next_elem = (2 * (cur[0] - x + 1), 2 * (cur[1] - y + 1))
			print "running:", next_elem[0], next_elem[1]
			if next_elem in path:
				continue
			path[next_elem] = path[cur] + [(x,y)]
			if next_elem == goal:
				print "Yippee:", path[next_elem]
				sys.exit(0)
			if next_elem[0] > goal[0] or next_elem[1] >goal[1]:
				continue
			q.put(next_elem)
