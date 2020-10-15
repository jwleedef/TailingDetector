import torch
import random
import argparse

def markingTargetGrid(gridMap, sequence, x, y, marking_value, marking_size):
    if (marking_size % 2) != 1:
        marking_size -= 1

    pad = int(marking_size / 2)
    for i in range(marking_size):
        for j in range(marking_size):
            x_var, y_var = x+i-pad, y+j-pad
            if x_var >= gridMap.shape[1]:
                x_var = gridMap.shape[1] - 1
            if x_var < 0:
                x_var = 0
            if y_var >= gridMap.shape[2]:
                y_var = gridMap.shape[2] - 1
            if y_var < 0:
                y_var = 0
                
            gridMap[sequence, x_var, y_var] += marking_value * (sequence + 1)

def markingSelectedDirection(direction, gridMap, init_x, init_y, horizontal, vertical, valid):
    if direction > 3:
        direction = random.randint(1, 3)

    markSize = 3

    if direction == 1:
        for s in range(10):
            x1, x2 = init_x[0] + (s * horizontal), init_x[1] + (s * horizontal)
            y1, y2 = init_y[0], init_y[1]
            markingTargetGrid(gridMap, s, x1, y1, markValue, markSize)
            if valid == True:
                markingTargetGrid(gridMap, s, x2, y2, markValue, markSize)                        
    elif direction == 2:
        for s in range(10):
            x1, x2 = init_x[0], init_x[1]
            y1, y2 = init_y[0] + (s * vertical), init_y[1] + (s * vertical)
            markingTargetGrid(gridMap, s, x1, y1, markValue, markSize)
            if valid == True:
                markingTargetGrid(gridMap, s, x2, y2, markValue, markSize)
    elif direction == 3:
        for s in range(10):
            x1, x2 = init_x[0] + (s * horizontal), init_x[1] + (s * horizontal)
            y1, y2 = init_y[0] + (s * vertical), init_y[1] + (s * vertical)
            markingTargetGrid(gridMap, s, x1, y1, markValue, markSize)
            if valid == True:
                markingTargetGrid(gridMap, s, x2, y2, markValue, markSize) 

def makeSyntheticData(gridMap, quadrant, w, h, numberOfMaxPerson, markValue, valid):

    x_guide = [0, w/2, w-1]
    y_guide = [0, h/2, h-1]

    min, max = int((w/2)/10), int(w/10)

    if quadrant == 1:
        init_x = [random.randint(x_guide[1], x_guide[2]), random.randint(x_guide[1], x_guide[2])]
        init_y = [random.randint(y_guide[0], y_guide[1]), random.randint(y_guide[0], y_guide[1])]

        direction = random.randint(1, 3)
        horizontal, vertical = random.randint(-max, -min), random.randint(min, max)
        markingSelectedDirection(direction, gridMap, init_x, init_y, horizontal, vertical, valid)

    elif quadrant == 2:
        init_x = [random.randint(x_guide[0], x_guide[1]), random.randint(x_guide[0], x_guide[1])]
        init_y = [random.randint(y_guide[0], y_guide[1]), random.randint(y_guide[0], y_guide[1])]

        direction = random.randint(1, 3)
        horizontal, vertical = random.randint(min, max), random.randint(min, max)
        markingSelectedDirection(direction, gridMap, init_x, init_y, horizontal, vertical, valid)

    elif quadrant == 3:
        init_x = [random.randint(x_guide[0], x_guide[1]), random.randint(x_guide[0], x_guide[1])]
        init_y = [random.randint(y_guide[1], y_guide[2]), random.randint(y_guide[1], y_guide[2])]

        direction = random.randint(1, 3)
        horizontal, vertical = random.randint(min, max), random.randint(-max, -min)
        markingSelectedDirection(direction, gridMap, init_x, init_y, horizontal, vertical, valid)

    elif quadrant == 4:
        init_x = [random.randint(x_guide[1], x_guide[2]), random.randint(x_guide[1], x_guide[2])]
        init_y = [random.randint(y_guide[1], y_guide[2]), random.randint(y_guide[1], y_guide[2])]
    
        direction = random.randint(1, 3)
        horizontal, vertical = random.randint(-max, -min), random.randint(-max, -min)
        markingSelectedDirection(direction, gridMap, init_x, init_y, horizontal, vertical, valid)
    
    # if numberOfMaxPerson > 2:
    #     for s in range(10):
    #         otherPerson = random.randint(0, numberOfMaxPerson - 2)
    #         for n in range(otherPerson):
    #             x = random.randint(0, w-1)
    #             y = random.randint(0, h-1)
    #             gridMap[s, x, y] += markValue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type=str, default='valid', help='type of dataset - valid: Tailing, invalid: NonTailing(Random Coordinates)')
    parser.add_argument('-p', '--numberOfMaxPerson', type=int, default=5, help='number of max person who is placed each frame')
    parser.add_argument('-n', '--numberOfData', type=int, default=10000, help='number of synthetic data to be created')
    parser.add_argument('-v', '--marking_value', type=int, default=1, help='values to mark on each grid-cell')
    parser.add_argument('--width', type=int, default=224, help='width size about each frame')
    parser.add_argument('--height', type=int, default=224, help='height size about each frame')
    parser.add_argument('--path', type=str, default='train', help='Dataset Path')
    args = parser.parse_args()

    type = args.type
    numberOfMaxPerson = args.numberOfMaxPerson
    numberOfData = args.numberOfData
    markValue = args.marking_value
    w = args.width
    h = args.height
    savePath = args.path

    gridMap = torch.zeros(10, w, h)
    unionGridMap = torch.zeros(w, h)

    quadrant = random.randint(1, 4)
    subQuadrant = random.randint(1, 4)
    while quadrant == subQuadrant :
        subQuadrant = random.randint(1, 4)

    if type == 'valid': # Tailing Situation
        valid = True
        if numberOfMaxPerson < 2:
            numberOfMaxPerson = 2

        for i in range(numberOfData):
            makeSyntheticData(gridMap, quadrant, w, h, numberOfMaxPerson, markValue, valid)
            datasetName = f'{savePath}/0/tailing-0-{i}.pt'
            for j in range(10):
                unionGridMap += gridMap[j,:]
            unionGridMap = torch.unsqueeze(unionGridMap, 0)
            torch.save(unionGridMap, datasetName)
            gridMap = torch.zeros(10, w, h)
            unionGridMap = torch.zeros(w, h)
            print(f'[Created] {i+1}th data')
        print(f'[DONE] Created {numberOfData} Tailing Data\n')

    elif type == 'invalid': # Non-Tailing situation    
        valid = False
        if numberOfMaxPerson < 1:
            numberOfMaxPerson = 1
        
        for i in range(numberOfData):
            makeSyntheticData(gridMap, quadrant, w, h, numberOfMaxPerson, markValue, valid)
            extra = random.randint(0, 1)
            if extra == 1 :
                makeSyntheticData(gridMap, subQuadrant, w, h, numberOfMaxPerson, markValue, valid)
            datasetName = f'{savePath}/1/non-tailing-1-{i}.pt'
            for j in range(10):
                unionGridMap += gridMap[j,:]
            unionGridMap = torch.unsqueeze(unionGridMap, 0)
            torch.save(unionGridMap, datasetName)
            gridMap = torch.zeros(10, w, h)
            unionGridMap = torch.zeros(w, h)
            print(f'[Created] {i+1}th data')

        # for i in range(numberOfData):
        #     for s in range(10):
        #         randPerson = random.randint(0, numberOfMaxPerson)
        #         for n in range(randPerson):
        #             x = random.randint(0, w - 1)
        #             y = random.randint(0, h - 1)
        #             gridMap[s, x, y] = markValue
        #     datasetName = f'train/0/Train-0-{i}.pt'
        #     torch.save(gridMap, datasetName)

        print(f'[DONE] Created {numberOfData} Non-Tailing Data\n')
