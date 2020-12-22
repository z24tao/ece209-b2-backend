import cherrypy
import csv
import json
import numpy as np
import random

from sklearn.ensemble import RandomForestRegressor

max_dist = 100.0
model_accel = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
model_turn = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)

aggregate_count = 0
aggregate_max = 1000
aggregate_data = []
aggregate_accel = []
aggregate_turn = []


class Object:
    def __init__(self, top, bot, left, right):
        self.top = np.float32(top)
        self.bot = np.float32(bot)
        self.left = np.float32(left)
        self.right = np.float32(right)

    def dist(self, srcx, srcy, direction):
        srcx, srcy, direction = np.float32(srcx), np.float32(srcy), np.float32(direction)
        if direction == 0:
            if srcx >= self.right or srcy > self.top or srcy < self.bot:
                return max_dist
            return max(0, self.left - srcx)

        if direction == 90:
            if srcy >= self.top or srcx > self.right or srcx < self.left:
                return max_dist
            return max(0, self.bot - srcy)

        if direction == 180:
            if srcx <= self.left or srcy > self.top or srcy < self.bot:
                return max_dist
            return max(0, srcx - self.right)

        if direction == 270:
            if srcy <= self.bot or srcx > self.right or srcx < self.left:
                return max_dist
            return max(0, srcy - self.top)

        rad = direction * np.pi / 180
        if 0 < direction < 90:
            if self.right <= srcx or self.top <= srcy:
                return max_dist
            if self.left <= srcx and self.bot <= srcy:
                return 0

            dist = max_dist
            if self.left > srcx:
                x_dist = self.left - srcx
                y_dist = x_dist * np.tan(rad)
                if self.bot <= srcy + y_dist <= self.top:
                    dist = min(dist, np.sqrt(x_dist * x_dist + y_dist * y_dist))

            if self.bot > srcy:
                y_dist = self.bot - srcy
                x_dist = y_dist / np.tan(rad)
                if self.left <= srcx + x_dist <= self.right:
                    dist = min(dist, np.sqrt(x_dist * x_dist + y_dist * y_dist))
            return dist

        if 90 < direction < 180:
            if self.left >= srcx or self.top <= srcy:
                return max_dist
            if self.right >= srcx and self.bot <= srcy:
                return 0

            dist = max_dist
            if self.right < srcx:
                x_dist = srcx - self.right
                y_dist = -1 * x_dist * np.tan(rad)
                if self.bot <= srcy + y_dist <= self.top:
                    dist = min(dist, np.sqrt(x_dist * x_dist + y_dist * y_dist))

            if self.bot > srcy:
                y_dist = self.bot - srcy
                x_dist = -1 * y_dist / np.tan(rad)
                if self.left <= srcx - x_dist <= self.right:
                    dist = min(dist, np.sqrt(x_dist * x_dist + y_dist * y_dist))
            return dist

        if 180 < direction < 270:
            if self.left >= srcx or self.bot >= srcy:
                return max_dist
            if self.right >= srcx and self.top >= srcy:
                return 0

            dist = max_dist
            if self.right < srcx:
                x_dist = srcx - self.right
                y_dist = x_dist * np.tan(rad)
                if self.bot <= srcy - y_dist <= self.top:
                    dist = min(dist, np.sqrt(x_dist * x_dist + y_dist * y_dist))

            if self.top < srcy:
                y_dist = srcy - self.top
                x_dist = y_dist / np.tan(rad)
                if self.left <= srcx - x_dist <= self.right:
                    dist = min(dist, np.sqrt(x_dist * x_dist + y_dist * y_dist))
            return dist

        if 270 < direction < 360:
            if self.right <= srcx or self.bot >= srcy:
                return max_dist
            if self.left <= srcx and self.top >= srcy:
                return 0

            dist = max_dist
            if self.left > srcx:
                x_dist = self.left - srcx
                y_dist = -1 * x_dist * np.tan(rad)
                if self.bot <= srcy - y_dist <= self.top:
                    dist = min(dist, np.sqrt(x_dist * x_dist + y_dist * y_dist))

            if self.top < srcy:
                y_dist = srcy - self.top
                x_dist = y_dist / np.tan(rad)
                if self.left <= srcx + x_dist <= self.right:
                    dist = min(dist, np.sqrt(x_dist * x_dist + y_dist * y_dist))
            return dist

        return max_dist


def min_dist(srcx, srcy, direction, objs):
    dist = max_dist
    for obj in objs:
        dist = min(dist, obj.dist(srcx, srcy, direction))
    return dist


def d2r(degree):
    return degree * np.pi / 180


def calc_data(srcx, srcy, direction, ctrl, goalx, goaly, objs):
    # print(srcx, srcy, direction, goalx, goaly, objs)
    right_top_x = srcx + 5.5 * np.cos(d2r(direction)) + 3.0 * np.cos(d2r(direction + 90))
    right_bot_x = srcx + 5.5 * np.cos(d2r(direction)) + 3.0 * np.cos(d2r(direction - 90))
    left_top_x = srcx + 5.5 * np.cos(d2r(direction + 180)) + 3.0 * np.cos(d2r(direction + 90))
    left_bot_x = srcx + 5.5 * np.cos(d2r(direction + 180)) + 3.0 * np.cos(d2r(direction - 90))
    right_top_y = srcy + 5.5 * np.sin(d2r(direction)) + 3.0 * np.sin(d2r(direction + 90))
    right_bot_y = srcy + 5.5 * np.sin(d2r(direction)) + 3.0 * np.sin(d2r(direction - 90))
    left_top_y = srcy + 5.5 * np.sin(d2r(direction + 180)) + 3.0 * np.sin(d2r(direction + 90))
    left_bot_y = srcy + 5.5 * np.sin(d2r(direction + 180)) + 3.0 * np.sin(d2r(direction - 90))
    # print(right_top_x, right_top_y, right_bot_x, right_bot_y)
    # print(left_top_x, left_top_y, left_bot_x, left_bot_y)
    return [
        min_dist((right_top_x + right_bot_x) / 2, (right_top_y + right_bot_y) / 2, direction % 360, objs),
        min_dist(right_top_x, right_top_y, (direction + 45) % 360, objs),
        min_dist((left_top_x + right_top_x) / 2, (left_top_y + right_top_y) / 2, (direction + 90) % 360, objs),
        min_dist(left_top_x, left_top_y, (direction + 135) % 360, objs),
        min_dist((left_top_x + left_bot_x) / 2, (left_top_y + left_bot_y) / 2, (direction + 180) % 360, objs),
        min_dist(left_bot_x, left_bot_y, (direction + 225) % 360, objs),
        min_dist((left_bot_x + right_bot_x) / 2, (left_bot_y + right_bot_y) / 2, (direction + 270) % 360, objs),
        min_dist(right_bot_x, right_bot_y, (direction + 315) % 360, objs),
        goalx - srcx,
        goaly - srcy,
        direction,
        ctrl
    ]


def calc_action(data_row):
    accel = model_accel.predict([data_row])[0]
    turn = model_turn.predict([data_row])[0]
    print(data_row, accel, turn)
    return ("+" if accel > 0.1 else "-" if accel < 0.1 else "0") + ("+" if turn > 0.1 else "-" if turn < 0.1 else "0")


def action_to_val(action):
    return [1 if action[0] == '+' else 0 if action[0] == '0' else -1,
            1 if action[1] == '+' else 0 if action[1] == '0' else -1]


def update_model():
    global aggregate_data
    global aggregate_accel
    global aggregate_turn
    global model_accel
    global model_turn
    model_accel = RandomForestRegressor(max_depth=2, random_state=0)
    model_accel.fit(aggregate_data, aggregate_accel)
    model_turn = RandomForestRegressor(max_depth=2, random_state=0)
    model_turn.fit(aggregate_data, aggregate_turn)
    aggregate_data = []
    aggregate_accel = []
    aggregate_turn = []


class Distance:
    exposed = True

    def POST(self, **kwargs):
        srcx, srcy = float(kwargs['srcx']), float(kwargs['srcy'])
        direction, wheel = float(kwargs['dir']), float(kwargs['ctrl'])
        data = json.loads(cherrypy.request.body.read())
        objs = [Object(item['top'], item['bot'], item['left'], item['right']) for item in data['obstacles']]
        goalx = (data['goal']['left'] + data['goal']['right']) / 2
        goaly = (data['goal']['top'] + data['goal']['bot']) / 2
        data_row = calc_data(srcx, srcy, direction, wheel, goalx, goaly, objs)
        action = calc_action(data_row)
        cherrypy.request.close()
        # num_action = action_to_val(action)
        # global aggregate_count
        # global aggregate_data
        # global aggregate_accel
        # global aggregate_turn
        # aggregate_data.append(data_row)
        # aggregate_accel.append(num_action[0])
        # aggregate_turn.append(num_action[1])
        # aggregate_count += 1
        # if aggregate_count == aggregate_max:
        #     aggregate_count = 0
        #     update_model()
        # print(aggregate_count)
        return action


class Train:
    exposed = True

    def POST(self, **kwargs):
        srcx, srcy = float(kwargs['srcx']), float(kwargs['srcy'])
        direction, wheel = float(kwargs['dir']), float(kwargs['ctrl'])
        data = json.loads(cherrypy.request.body.read())
        objs = [Object(item['top'], item['bot'], item['left'], item['right']) for item in data['obstacles']]
        goalx = (data['goal']['left'] + data['goal']['right']) / 2
        goaly = (data['goal']['top'] + data['goal']['bot']) / 2
        data_row = calc_data(srcx, srcy, direction, wheel, goalx, goaly, objs)
        data_accel = data['ctrl1']
        data_turn = data['ctrl2']

        global aggregate_data
        global aggregate_accel
        global aggregate_turn
        aggregate_data.append(data_row)
        aggregate_accel.append(data_accel)
        aggregate_turn.append(data_turn)
        return ""


class Init:
    exposed = True

    def POST(self):
        global model_accel
        global model_turn
        global aggregate_data
        global aggregate_accel
        global aggregate_turn
        print(len(aggregate_data), "new rows")
        data_file = open('data.csv', 'a')
        accel_file = open('accel.csv', 'a')
        turn_file = open('turn.csv', 'a')
        data_writer = csv.writer(data_file)
        accel_writer = csv.writer(accel_file)
        turn_writer = csv.writer(turn_file)
        for data_row, accel_row, turn_row in zip(aggregate_data, aggregate_accel, aggregate_turn):
            data_writer.writerow([round(x, 2) for x in data_row])
            accel_writer.writerow([accel_row])
            turn_writer.writerow([turn_row])
        data_file.close()
        accel_file.close()
        turn_file.close()
        aggregate_data = []
        aggregate_accel = []
        aggregate_turn = []
        data_reader = csv.reader(open('data.csv'))
        accel_reader = csv.reader(open('accel.csv'))
        turn_reader = csv.reader(open('turn.csv'))
        for data_row in data_reader:
            aggregate_data.append([float(i) for i in data_row])
        for accel_row in accel_reader:
            aggregate_accel.append(float(accel_row[0]))
        for turn_row in turn_reader:
            aggregate_turn.append(float(turn_row[0]))
        print("total rows", len(aggregate_data), len(aggregate_accel), len(aggregate_turn))
        model_accel = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
        model_accel.fit(aggregate_data, aggregate_accel)
        model_turn = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
        model_turn.fit(aggregate_data, aggregate_turn)
        aggregate_data = []
        aggregate_accel = []
        aggregate_turn = []
        cherrypy.request.close()
        return ""


def init_model():
    global model_accel
    global model_turn
    size = 100
    rand_data = []
    rand_accel = random.choices([-1., 0., 1.], k=size)
    rand_turn = random.choices([-1., 0., 1.], k=size)
    for i in range(size):
        rand_data.append([random.random() * 5 for _ in range(8)] + [random.random() * 20 for _ in range(2)])
    model_accel = RandomForestRegressor(max_depth=2, random_state=0)
    model_accel.fit(rand_data, rand_accel)
    model_turn = RandomForestRegressor(max_depth=2, random_state=0)
    model_turn.fit(rand_data, rand_turn)


car_speed = 10
max_dir = 45
concern_dist = 10.0


def auto_train(iters):
    car = Car()
    car.set_goal(50, 0, 15, 10)
    car.add_obstacle(20, 0, 5, 30)
    car.add_obstacle(-30, 0, 5, 100)
    car.add_obstacle(0, 50, 120, 5)
    car.add_obstacle(0, -50, 120, 5)
    car.add_obstacle(60, 0, 5, 100)

    data = []
    accels = []
    turns = []

    for i in range(iters):
        start = [
            random.random() * 360,
            random.random() * 50,
            random.random() * 50 - 25,
            0
        ]
        # start = [90, 0, 0, 0]
        data_row = calc_data(car.pos[0], car.pos[1], (car.angle + 270) % 360, car.ctrl_dir, car.goal[0], car.goal[1], car.get_objs())
        while True:
            car.set_self(start[0], start[1], start[2], start[3])
            all_positive = True
            for j in range(8):
                if data_row[j] == 0:
                    all_positive = False
            if all_positive:
                break

            start = [
                (random.random() * 180 + 270) % 360,
                random.random() * 50,
                random.random() * 50 - 25,
                0
            ]
            # start = [90, 0, 0, 0]
            data_row = calc_data(car.pos[0], car.pos[1], (car.angle + 270) % 360, car.ctrl_dir, car.goal[0], car.goal[1], car.get_objs())

        best = [0, 0]
        best_score = -1000000
        for accel in [1]:
            for turn in [-1, 0, 1]:
                car.set_self(start[0], start[1], start[2], start[3])
                car.control(accel, turn)
                for frame in range(10):
                    car.drive()
                score = car.score()
                if score > best_score:
                    best_score = score
                    best = [accel, turn]
        if i % 100 == 0:
            print(i, "score", best_score, "ctrl", best, "setting", start)

        if best_score > -2000:
            data.append(data_row)
            accels.append(best[0])
            turns.append(best[1])

    data_file = open('data.csv', 'w')
    accel_file = open('accel.csv', 'w')
    turn_file = open('turn.csv', 'w')
    data_writer = csv.writer(data_file)
    accel_writer = csv.writer(accel_file)
    turn_writer = csv.writer(turn_file)
    for data_row, accel_row, turn_row in zip(data, accels, turns):
        data_writer.writerow([round(x, 2) for x in data_row])
        accel_writer.writerow([accel_row])
        turn_writer.writerow([turn_row])
    data_file.close()
    accel_file.close()
    turn_file.close()


def grid_train():
    car = Car()
    car.set_goal(50, 0, 15, 10)
    car.add_obstacle(20, 0, 5, 30)
    # car.add_obstacle(-30, 0, 5, 100)
    # car.add_obstacle(0, 50, 120, 5)
    # car.add_obstacle(0, -50, 120, 5)
    # car.add_obstacle(60, 0, 5, 100)
    # car.add_obstacle(50, 15, 15, 10)
    # car.add_obstacle(50, -15, 15, 10)

    data = []
    accels = []
    turns = []

    for x in range(-25, 75, 2):
        print(x)
        for z in range(-50, 50, 2):
            for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
                start = [angle, x, z, 0]
                car.set_self(start[0], start[1], start[2], start[3])
                data_row = calc_data(car.pos[0], car.pos[1], (car.angle + 270) % 360, car.ctrl_dir, car.goal[0],
                                     car.goal[1], car.get_objs())
                best = [0, 0]
                best_score = -1000000
                for accel in [1]:
                    for turn in [-1, 0, 1]:
                        car.set_self(start[0], start[1], start[2], start[3])
                        for frame in range(10):
                            car.control(accel, turn)
                            car.drive()
                        score = car.score()
                        if score > best_score:
                            best_score = score
                            best = [accel, turn]

                if best_score > -2000:
                    data.append(data_row)
                    accels.append(best[0])
                    turns.append(best[1])

    data_file = open('data.csv', 'w')
    accel_file = open('accel.csv', 'w')
    turn_file = open('turn.csv', 'w')
    data_writer = csv.writer(data_file)
    accel_writer = csv.writer(accel_file)
    turn_writer = csv.writer(turn_file)
    for data_row, accel_row, turn_row in zip(data, accels, turns):
        data_writer.writerow([round(x, 2) for x in data_row])
        accel_writer.writerow([accel_row])
        turn_writer.writerow([turn_row])
    data_file.close()
    accel_file.close()
    turn_file.close()


class Car:
    def __init__(self):
        self.angle = 0.0
        self.pos = [0.0, 0.0]
        self.locked = False

        self.goal = [0.0, 0.0, 0.0, 0.0]
        self.obstacles = []

        self.ctrl_dir = 0.0
        self.speed = 0.0

    def set_self(self, angle, posx, posy, speed):
        self.angle = angle
        self.pos = [posx, posy]
        self.locked = False
        self.speed = speed
        self.ctrl_dir = angle

    def set_goal(self, posx, posy, dimx, dimy):
        self.goal = [posx, posy, dimx, dimy]

    def clear_obstacles(self):
        self.obstacles = []

    def add_obstacle(self, posx, posy, dimx, dimy):
        self.obstacles.append([posx, posy, dimx, dimy])

    def get_objs(self):
        objs = []
        for obj in self.obstacles:
            objs.append(Object(obj[1] + obj[3] / 2, obj[1] - obj[3] / 2, obj[0] - obj[2] / 2, obj[0] + obj[2] / 2))
        return objs

    def score(self):
        score = 0
        data_row = calc_data(self.pos[0], self.pos[1], (self.angle + 270) % 360, self.ctrl_dir,
                             self.goal[0], self.goal[1], self.get_objs())
        # print("datarow", data_row)
        for i in data_row[:8]:
            if i < 1:
                return -1000000

        score -= np.sqrt(abs(data_row[8]) ** 2 + abs(data_row[9]) ** 2) * 15  # goal dist

        # print("goal", score)

        score -= min(data_row[10], 360 - data_row[10]) * 0.1  # goal angle diff
        # print("angle", score)

        # print(data_row[0], data_row[4], min(concern_dist, data_row[0]), min(concern_dist, data_row[4]))
        score -= max(0.0, 1000.0 / min(concern_dist, data_row[0])) * 3  # front and back obstacle dist
        score -= max(0.0, 1000.0 / min(concern_dist, data_row[4]))  # * 3
        # print("frontback", score)

        score -= max(0.0, 1000.0 / min(concern_dist, data_row[1]))  # other obstacle dist
        score -= max(0.0, 1000.0 / min(concern_dist, data_row[2]))
        score -= max(0.0, 1000.0 / min(concern_dist, data_row[3]))
        score -= max(0.0, 1000.0 / min(concern_dist, data_row[5]))
        score -= max(0.0, 1000.0 / min(concern_dist, data_row[6]))
        score -= max(0.0, 1000.0 / min(concern_dist, data_row[7]))
        # print("other", score)

        return score

    def control(self, accel, turn):
        # print("control start", self.ctrl_dir, self.angle)
        if accel == 1:
            self.speed = car_speed
        elif accel == -1:
            self.speed = -car_speed

        if turn == -1:
            self.locked = False
            self.ctrl_dir -= 5.0
            if self.ctrl_dir < 0:
                self.ctrl_dir += 360
        elif turn == 1:
            self.locked = False
            self.ctrl_dir += 5.0
            if self.ctrl_dir > 360:
                self.ctrl_dir -= 360

        car_dir = self.ctrl_dir
        if self.angle - car_dir > 180:
            self.ctrl_dir += 360
        elif car_dir - self.angle > 180:
            car_dir += 360

        if self.ctrl_dir > car_dir + max_dir:
            self.ctrl_dir = car_dir + max_dir
        elif self.ctrl_dir < car_dir - max_dir:
            self.ctrl_dir = car_dir - max_dir
        # print("control", accel, turn, self.ctrl_dir, self.angle)

    def drive(self):
        car_dir = self.angle
        car_deg = car_dir / 180 * np.pi
        wheel_dir = self.ctrl_dir
        if self.locked:
            wheel_dir = car_dir

        front_pos = [self.pos[0] + np.sin(car_deg) * 3, self.pos[1] + np.cos(car_deg) * 3]
        back_pos = [self.pos[0] - np.sin(car_deg) * 3, self.pos[1] - np.cos(car_deg) * 3]

        speed = self.speed / car_speed / 5
        wheel_deg = wheel_dir / 180 * np.pi
        front_pos = [front_pos[0] + np.sin(wheel_deg) * speed, front_pos[1] + np.cos(wheel_deg) * speed]
        tan = (front_pos[0] - back_pos[0]) / (front_pos[1] - back_pos[1])
        turn = np.arctan(tan)

        if front_pos[1] < back_pos[1]:
            turn += np.pi

        self.angle = turn * 180 / np.pi
        self.pos = [(front_pos[0] + back_pos[0]) / 2, (front_pos[1] + back_pos[1]) / 2]

        new_car_dir = self.angle
        if new_car_dir - wheel_dir > 180:
            wheel_dir += 360
        elif wheel_dir - new_car_dir > 180:
            new_car_dir += 360

        if -10 < wheel_dir - new_car_dir < 10:
            self.locked = True

        # print("drive", self.pos, self.angle)


if __name__ == '__main__':
    grid_train()
    data_reader = csv.reader(open('data.csv'))
    accel_reader = csv.reader(open('accel.csv'))
    turn_reader = csv.reader(open('turn.csv'))
    in_data, in_accel, in_turn = [], [], []
    for data_row in data_reader:
        in_data.append([float(i) for i in data_row])
    for accel_row in accel_reader:
        in_accel.append(float(accel_row[0]))
    for turn_row in turn_reader:
        in_turn.append(float(turn_row[0]))
    print("total rows", len(in_data))
    model_accel.fit(in_data, in_accel)
    model_turn.fit(in_data, in_turn)

    # car = Car()
    # car.set_goal(50, 0, 15, 10)
    # car.add_obstacle(20, 0, 5, 30)
    # car.add_obstacle(-30, 0, 5, 100)
    # car.add_obstacle(0, 50, 120, 5)
    # car.add_obstacle(0, -50, 120, 5)
    # car.add_obstacle(60, 0, 5, 100)
    # start = [90, 5, 0, 0]
    # car.set_self(start[0], start[1], start[2], start[3])
    # print(car.score())
    #
    # best = [0, 0]
    # best_score = -1000000
    # for accel in [1]:
    #     for turn in [-1, 0, 1]:
    #         car.set_self(start[0], start[1], start[2], start[3])
    #         car.control(accel, turn)
    #         for frame in range(10):
    #             car.drive()
    #         score = car.score()
    #         if score > best_score:
    #             best_score = score
    #             best = [accel, turn]
    #         print(accel, turn, score, car.pos, car.angle)

    # auto_train(10000)

    # init_model()
    cherrypy.tree.mount(
        Distance(), '/distance', {
            '/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}
        }
    )
    cherrypy.tree.mount(
        Train(), '/train', {
            '/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}
        }
    )
    cherrypy.tree.mount(
        Init(), '/init', {
            '/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}
        }
    )

    cherrypy.engine.start()
    cherrypy.engine.block()
