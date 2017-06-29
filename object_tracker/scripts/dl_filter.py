#!/usr/bin/env python
import numpy as np

SPEED_LIMIT = 16.67 # 60 km/h in m/s
INIT_TIME = 1.0 # sec
RESET_TIME = 1.0 # sec

class dl_filter:

	def __init__(self):
		self.init_start_time = -1
		self.reset_start_time = -1
		self.initialized = False
		self.valid = False

		self.prev_time = 0.0
		self.prev_vel = [0, 0]
		self.prev_box = []

	def filter_by_velocity(self, boxes, ts_sec, ts_nsec):
		time = to_time(ts_sec, ts_nsec)
		output = []
		if len(boxes) < 1: # no box
			if not self.initialized:
				# doing nothing for now
				a = 0
			else:
				box = np.copy(self.prev_box)
				dp = self.prev_vel * (time - self.prev_time)
				box[:,:,0] += dp[0]
				box[:,:,1] += dp[1]
				output.append(box)
		else: # filtered box exists
			if (not self.initialized) or (not self.valid):
				# start init timer
				if self.init_start_time < 0:
					# save current info
					box = []
					if len(boxes) == 1:
						box = boxes[0]
					else:
						box = self.select_box(boxes)
					output.append(box)
					prev_vel = [0, 0]
					prev_time = time
					prev_box = box
					init_start_time = time
				# init timer running
				else:
					#filter with velocity
					found = []
					for box in boxes:
						point = center(box)[:2]
						vel = self.velocity(point, time)
						if norm(vel) < SPEED_LIMIT:
							found.append(box)

					if len(found) < 1:
						# reset init timer
						self.init_start_time = -1
						#predict
						if not self.initialized:
							# do nothing for now
							a = 0
						else:
							box = np.copy(self.prev_box)
							dp = self.prev_vel * (time - self.prev_time)
							box[:,:,0] += dp[0]
							box[:,:,1] += dp[1]
							output.append(box)
					else: # found
						# save current info
						box = []
						if len(boxes) == 1:
							box = boxes[0]
						else:
							box = self.select_box(boxes)
						output.append(box)
						point = center(box)[:2]
						vel = velocity(point, time)
						self.prev_vel = vel
						self.prev_time = time
						self.prev_box = box
						# if init condition satisfied
						if time - self.init_start_time > INIT_TIME:
							self.initialized = True
							self.valid = True
							self.init_start_time = -1
			else: # initialized and valid
				# filter with velocity
				found = []
				for box in boxes:
					point = center(box)[:2]
					vel = self.velocity(point, time)
					if norm(vel) < SPEED_LIMIT:
						found.append(box)

				if len(found) < 1:
					box = np.copy(self.prev_box)
					dp = self.prev_vel * (time - self.prev_time)
					box[:,:,0] += dp[0]
					box[:,:,1] += dp[1]
					output.append(box)
				else: # found
					# save current info
					box = []
					if len(boxes) == 1:
						box = boxes[0]
					else:
						box = self.select_box(boxes)
					output.append(box)
					point = center(box)[:2]
					vel = velocity(point, time)
					self.prev_vel = vel
					self.prev_time = time
					self.prev_box = box

				# start reset timer
				if self.reset_start_time < 0:
					self.reset_start_time = time
				elif time - self.reset_start_time > RESET_TIME:
					self.valid = False
					self.reset_start_time = -1
					# start initialization
					self.filter_by_velocity(boxes, ts_sec, ts_nsec)
		return output

	def velocity(self, pos, time):
		prev_pos = center(self.prev_box)[:2]
		return float(pos - prev_pos) / (time - self.prev_time)

	def select_box(self, boxes):
		# TODO: need to fix this
		return boxes[0]

def to_time(ts_sec, ts_nsec):
	return 1E-9 * float(ts_nsec) + ts_sec

def center(box):
	return 0.5 * (box[0] + box[6])

def norm(vec):
	return np.sqrt(vec[0]*vec[0] + vec[1]*vec[1])