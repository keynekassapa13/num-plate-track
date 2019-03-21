import numpy as np

files = [
    [{'name': 'car1.jpg', 'pm_thresh': 0.58, 'type': 'white'}],
    [{'name': 'car2.jpg', 'pm_thresh': 0.55, 'type': 'special', 'option':{
        'type': 'color',
        'lower_white': np.array([0,0,210], dtype=np.uint8),
        'upper_white': np.array([255,30,255], dtype=np.uint8),
        'no_bitwise': False
    }}],
    [{'name': 'car3.jpg', 'pm_thresh': 0.42, 'type': 'yellow'}],
    [{'name': 'car4.jpg', 'pm_thresh': 0.5, 'type': 'white_bg'}],
    [{'name': 'car5.jpg', 'pm_thresh': 0.15, 'type': 'yellow'}],
    [{'name': 'car6.jpg', 'pm_thresh': 0.17, 'type': 'white'}],
    [{'name': 'car7.jpg', 'pm_thresh': 0.28, 'type': 'white'}],
    [{'name': 'car8.jpg', 'pm_thresh': 0.19, 'type': 'white'}],
    [{'name': 'car9.jpg', 'pm_thresh': 0.56, 'type': 'yellow'}],
    [{'name': 'car10.jpg', 'pm_thresh': 0.40, 'type': 'white'}],
    [{'name': 'car11.jpg', 'pm_thresh': 0.40, 'type': 'special', 'option': {
        'type': 'nope',
        'no_bitwise': False
    }}],
    [{'name': 'car12.jpg', 'pm_thresh': 0.40, 'type': 'white'}],
    [{'name': 'car13.jpg', 'pm_thresh': 0.40, 'type': 'special', 'option': {
        'type': 'nope',
        'no_bitwise': False
    }}],
    [{'name': 'car14.jpg', 'pm_thresh': 0.45, 'type': 'white'}],
    [{'name': 'car15.jpg', 'pm_thresh': 0.6, 'type': 'special', 'option': {
        'type': 'color',
        'lower_white': np.array([0, 150, 150], dtype=np.uint8),
        'upper_white': np.array([40, 255, 205], dtype=np.uint8),
        'no_bitwise': False
    }}],
    [{'name': 'car16.jpg', 'pm_thresh': 0.20, 'type': 'white'}],
    [{'name': 'car17.jpg', 'pm_thresh': 0.20, 'type': 'white'}],
    [{'name': 'car18.jpg', 'pm_thresh': 0.25, 'type': 'special', 'option': {
        'type': 'color',
        'lower_white': np.array([0, 0, 0], dtype=np.uint8),
        'upper_white': np.array([180, 255, 48], dtype=np.uint8),
        'no_bitwise': True
    }}],
    [{'name': 'car19.jpg', 'pm_thresh': 0.40, 'type': 'yellow'}],
    [{'name': 'car20.jpg', 'pm_thresh': 0.33, 'type': 'yellow'}],
    [{'name': 'car21.jpg', 'pm_thresh': 0.60, 'type': 'yellow'}],
    [{'name': 'car22.jpg', 'pm_thresh': 0.50, 'type': 'white_bg'}],
    [{'name': 'car23.jpg', 'pm_thresh': 0.20, 'type': 'special', 'option': {
        'type': 'color',
        'lower_white': np.array([0, 0, 0], dtype=np.uint8),
        'upper_white': np.array([180, 255, 200], dtype=np.uint8),
        'no_bitwise': True
    }}],
    [{'name': 'car24.jpg', 'pm_thresh': 0.20, 'type': 'white'}],
    [{'name': 'car25.jpg', 'pm_thresh': 0.62, 'type': 'special', 'option': {
        'type': 'color',
        'lower_white': np.array([0,0,230], dtype=np.uint8),
        'upper_white': np.array([255,20,255], dtype=np.uint8),
        'no_bitwise': False
    }}],
    [{'name': 'car26.jpg', 'pm_thresh': 0.60, 'type': 'special', 'option': {
        'type': 'nope',
        'no_bitwise': False
    }}],
]

temp_files = [
    'G',
    'K',
    'Z',
    '6',
    'D',
    '0',
    'C',
    '1',
    '2',
    '3',
    '4',
    '5',
    '7',
    '8',
    '9',
    'A',
    'B',
    'E',
    'H',
    'L',
    'M',
    'R',
    'T',
    'W',
    'X'
]
