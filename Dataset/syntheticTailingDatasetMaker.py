import torch
import random
import argparse

def makeTailingSituation(quadrant, w, h, numberOfMaxPerson, markValue):

    gridMap = torch.zeros(10, w, h)

    x_guide = [0, w/2, w-1]
    y_guide = [0, h/2, h-1]

    if quadrant == 1:
        init_x = [random.randint(x_guide[1], x_guide[2]), random.randint(x_guide[1], x_guide[2])]
        init_y = [random.randint(y_guide[0], y_guide[1]), random.randint(y_guide[0], y_guide[1])]

        direction = random.randint(1, 3)
        horizontal, vertical = random.randint(-10, -5), random.randint(5, 10)

        if direction == 1:
            for s in range(10):
                gridMap[s, init_x[0] + (s * horizontal), init_y[0]] = markValue
                gridMap[s, init_x[1] + (s * horizontal), init_y[1]] = markValue
        elif direction == 2:
            for s in range(10):
                gridMap[s, init_x[0], init_y[0] + (s * vertical)] = markValue
                gridMap[s, init_x[1], init_y[1] + (s * vertical)] = markValue
        elif direction == 3:
            for s in range(10):
                gridMap[s, init_x[0] + (s * horizontal), init_y[0] + (s * vertical)] = markValue
                gridMap[s, init_x[1] + (s * horizontal), init_y[1] + (s * vertical)] = markValue

    elif quadrant == 2:
        init_x = [random.randint(x_guide[0], x_guide[1]), random.randint(x_guide[0], x_guide[1])]
        init_y = [random.randint(y_guide[0], y_guide[1]), random.randint(y_guide[0], y_guide[1])]

        direction = random.randint(1, 3)
        horizontal, vertical = random.randint(5, 10), random.randint(5, 10)

        if direction == 1:
            for s in range(10):
                gridMap[s, init_x[0] + (s * horizontal), init_y[0]] = markValue
                gridMap[s, init_x[1] + (s * horizontal), init_y[1]] = markValue
        elif direction == 2:
            for s in range(10):
                gridMap[s, init_x[0], init_y[0] + (s * vertical)] = markValue
                gridMap[s, init_x[1], init_y[1] + (s * vertical)] = markValue
        elif direction == 3:
            for s in range(10):
                gridMap[s, init_x[0] + (s * horizontal), init_y[0] + (s * vertical)] = markValue
                gridMap[s, init_x[1] + (s * horizontal), init_y[1] + (s * vertical)] = markValue

    elif quadrant == 3:
        init_x = [random.randint(x_guide[0], x_guide[1]), random.randint(x_guide[0], x_guide[1])]
        init_y = [random.randint(y_guide[1], y_guide[2]), random.randint(y_guide[1], y_guide[2])]

        direction = random.randint(1, 3)
        horizontal, vertical = random.randint(5, 10), random.randint(-10, -5)

        if direction == 1:
            for s in range(10):
                gridMap[s, init_x[0] + (s * horizontal), init_y[0]] = markValue
                gridMap[s, init_x[1] + (s * horizontal), init_y[1]] = markValue
        elif direction == 2:
            for s in range(10):
                gridMap[s, init_x[0], init_y[0] + (s * vertical)] = markValue
                gridMap[s, init_x[1], init_y[1] + (s * vertical)] = markValue
        elif direction == 3:
            for s in range(10):
                gridMap[s, init_x[0] + (s * horizontal), init_y[0] + (s * vertical)] = markValue
                gridMap[s, init_x[1] + (s * horizontal), init_y[1] + (s * vertical)] = markValue

    elif quadrant == 4:
        init_x = [random.randint(x_guide[1], x_guide[2]), random.randint(x_guide[1], x_guide[2])]
        init_y = [random.randint(y_guide[1], y_guide[2]), random.randint(y_guide[1], y_guide[2])]
    
        direction = random.randint(1, 3)
        horizontal, vertical = random.randint(-10, -5), random.randint(-10, -5)

        if direction == 1:
            for s in range(10):
                gridMap[s, init_x[0] + (s * horizontal), init_y[0]] = markValue
                gridMap[s, init_x[1] + (s * horizontal), init_y[1]] = markValue
        elif direction == 2:
            for s in range(10):
                gridMap[s, init_x[0], init_y[0] + (s * vertical)] = markValue
                gridMap[s, init_x[1], init_y[1] + (s * vertical)] = markValue
        elif direction == 3:
            for s in range(10):
                gridMap[s, init_x[0] + (s * horizontal), init_y[0] + (s * vertical)] = markValue
                gridMap[s, init_x[1] + (s * horizontal), init_y[1] + (s * vertical)] = markValue
    
    if numberOfMaxPerson > 2:
        for s in range(10):
            otherPerson = random.randint(0, numberOfMaxPerson - 2)
            for n in range(otherPerson):
                x = random.randint(0, w-1)
                y = random.randint(0, h-1)
                gridMap[s, x, y] = markValue

    return gridMap

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type=str, default='valid', help='type of dataset - valid: Tailing, invalid: NonTailing(Random Coordinates)')
    parser.add_argument('-p', '--numberOfMaxPerson', type=int, default=5, help='number of max person who is placed each frame')
    parser.add_argument('-n', '--numberOfData', type=int, default=10000, help='number of synthetic data to be created')
    parser.add_argument('-v', '--marking_value', type=int, default=1, help='values to mark on each grid-cell')
    parser.add_argument('--width', type=int, default=224, help='width size about each frame')
    parser.add_argument('--height', type=int, default=224, help='height size about each frame')
    args = parser.parse_args()

    type = args.type
    numberOfMaxPerson = args.numberOfMaxPerson
    numberOfData = args.numberOfData
    markValue = args.marking_value
    w = args.width
    h = args.height

    gridMap = torch.zeros(10, w, h)

    if type == 'valid': # Tailing Situation
        if numberOfMaxPerson < 2:
            numberOfMaxPerson = 2

        for i in range(numberOfData):
            quadrant = random.randint(1, 4)
            gridMap = makeTailingSituation(quadrant, w, h, numberOfMaxPerson, markValue)
            datasetName = f'train/1/Train-1-{i}.pt'
            torch.save(gridMap, datasetName)

        print("[DONE] Created {numberOfData} Tailing Simulation Data\n")

    elif type == 'invalid': # Non-Tailing situation    
        for i in range(numberOfData):
            for s in range(10):
                numOfPerson = random.randint(0, numberOfMaxPerson)
                for n in range(numOfPerson):
                    x = random.randint(0, w - 1)
                    y = random.randint(0, h - 1)
                    gridMap[s, x, y] = markValue
            datasetName = f'train/0/Train-0-{i}.pt'
            torch.save(gridMap, datasetName)

        print("[DONE] Created {numberOfData} invalid Data\n")
