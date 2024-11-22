# CLASSES = [
#     'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
#     'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
#     'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
#     'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
#     'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
#     'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
# ]


CLASSES =  ['Hamate', 'Scaphoid', 'Lunate', 'Trapezium', 'Capitate', 'Triquetrum', 'Trapezoid', 'Pisiform']
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}

IND2CLASS = {v: k for k, v in CLASS2IND.items()}

TRAIN_DATA_DIR = '/data/ephemeral/home/data/train'
TEST_DATA_DIR = '/data/ephemeral/home/data/test'

# Color palette for visualization
PALETTE = [
    (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]