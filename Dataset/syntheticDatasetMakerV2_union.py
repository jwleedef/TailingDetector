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
                
            gridMap[sequence, x_var, y_var] = marking_value

def markingSelectedDirection(direction, gridMap, init_x, init_y, horizontal, vertical, valid):
    if direction > 3:
        direction = random.randint(1, 3)

    markSize = 3
    randPoint = random.randint(-5, 5)

    if direction == 1:
        for s in range(10):
            x1 = init_x + (s * (horizontal + randPoint))
            y1 = init_y + (s * randPoint)
            markingTargetGrid(gridMap, s, x1, y1, markValue, markSize)
            if valid == True:
                x2 = x1 - 2 * horizontal + randPoint
                y2 = y1 + randPoint
                markingTargetGrid(gridMap, s, x2, y2, markValue, markSize)                        
    elif direction == 2:
        for s in range(10):
            x1 = init_x + (s * randPoint)
            y1 = init_y + (s * (vertical + randPoint))
            markingTargetGrid(gridMap, s, x1, y1, markValue, markSize)
            if valid == True:
                x2 = x1 + randPoint
                y2 = y1 - 2 * vertical + randPoint                
                markingTargetGrid(gridMap, s, x2, y2, markValue, markSize)
    elif direction == 3:
        for s in range(10):
            x1 = init_x + (s * (horizontal + randPoint))
            y1 = init_y + (s * (vertical + randPoint))
            markingTargetGrid(gridMap, s, x1, y1, markValue, markSize)
            if valid == True:
                x2 = x1 - 2 * horizontal + randPoint
                y2 = y1 - 2 * vertical + randPoint                
                markingTargetGrid(gridMap, s, x2, y2, markValue, markSize) 

def makeSyntheticData(gridMap, quadrant, w, h, markValue, valid):

    x_guide = [0, w/2, w-1]
    y_guide = [0, h/2, h-1]

    min, max = int(w/20), int(w/10)

    if quadrant == 1:
        init_x = random.randint(x_guide[1], x_guide[2])
        init_y = random.randint(y_guide[0], y_guide[1])

        direction = random.randint(1, 3)
        horizontal, vertical = random.randint(-max, -min), random.randint(min, max)
        markingSelectedDirection(direction, gridMap, init_x, init_y, horizontal, vertical, valid)

    elif quadrant == 2:
        init_x = random.randint(x_guide[0], x_guide[1])
        init_y = random.randint(y_guide[0], y_guide[1])

        direction = random.randint(1, 3)
        horizontal, vertical = random.randint(min, max), random.randint(min, max)
        markingSelectedDirection(direction, gridMap, init_x, init_y, horizontal, vertical, valid)

    elif quadrant == 3:
        init_x = random.randint(x_guide[0], x_guide[1])
        init_y = random.randint(y_guide[1], y_guide[2])

        direction = random.randint(1, 3)
        horizontal, vertical = random.randint(min, max), random.randint(-max, -min)
        markingSelectedDirection(direction, gridMap, init_x, init_y, horizontal, vertical, valid)

    elif quadrant == 4:
        init_x = random.randint(x_guide[1], x_guide[2])
        init_y = random.randint(y_guide[1], y_guide[2])
    
        direction = random.randint(1, 3)
        horizontal, vertical = random.randint(-max, -min), random.randint(-max, -min)
        markingSelectedDirection(direction, gridMap, init_x, init_y, horizontal, vertical, valid)
    
    if random.randint(0, 1):
        otherPerson = random.randint(0, 2)
        for n in range(otherPerson):
            x = random.randint(0, w-1)
            y = random.randint(0, h-1)
            for s in range(10):
                markingTargetGrid(gridMap, s, x, y, markValue, 3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type=str, default='valid', help='type of dataset - valid: Tailing, invalid: Non-Tailing')
    parser.add_argument('-n', '--numberOfData', type=int, default=10000, help='number of synthetic data to be created')
    parser.add_argument('-v', '--marking_value', type=int, default=1, help='values to mark on each grid-cell')
    parser.add_argument('--width', type=int, default=120, help='width size about each frame')
    parser.add_argument('--height', type=int, default=120, help='height size about each frame')
    parser.add_argument('--path', type=str, default='train', help='Dataset Path')
    args = parser.parse_args()

    type = args.type
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

        for i in range(numberOfData):
            makeSyntheticData(gridMap, quadrant, w, h, markValue, valid)
            datasetName = f'{savePath}/0/tailing-0-{i}.pt'
            for j in range(10):
                unionGridMap += gridMap[j,:]
            unionGridMap = torch.unsqueeze(unionGridMap, 0)
            torch.save(unionGridMap, datasetName)
            gridMap = torch.zeros(10, w, h)
            unionGridMap = torch.zeros(w, h)            
            # torch.save(gridMap, datasetName)
            # gridMap = torch.zeros(10, w, h)
            if (i+1) % 100 == 0:
                print(f'[Created] {i+1}th data')
        print(f'[DONE] Created {numberOfData} Tailing Data\n')

    elif type == 'invalid': # Non-Tailing situation    
        valid = False
        
        for i in range(numberOfData):
            makeSyntheticData(gridMap, quadrant, w, h, markValue, valid)
            extra = random.randint(0, 1)
            if extra == 1 :
                makeSyntheticData(gridMap, subQuadrant, w, h, markValue, valid)
            datasetName = f'{savePath}/1/non-tailing-1-{i}.pt'
            for j in range(10):
                unionGridMap += gridMap[j,:]
            unionGridMap = torch.unsqueeze(unionGridMap, 0)
            torch.save(unionGridMap, datasetName)
            gridMap = torch.zeros(10, w, h)
            unionGridMap = torch.zeros(w, h)            
            # torch.save(gridMap, datasetName)
            # gridMap = torch.zeros(10, w, h)
            if (i+1) % 100 == 0:
                print(f'[Created] {i+1}th data')
        print(f'[DONE] Created {numberOfData} Non-Tailing Data\n')
