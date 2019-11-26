workdir  = 'outputs/model002'
seed = 2050
apex = False

n_fold = 15
epoch =300
resume_from = None

batch_size = 10
num_workers = 20
# imgsize = (320, 640) #(height, width)
early_stop = 10

apex = False
gpu=[0, 1]

# path
train_csv = 'data/train.csv'
test_csv = 'data/sample_submission.csv'
train_images = 'data/train_images'
test_images = 'data/test_images'

model = dict(
    name='efficientnet-b0',
    params=dict(
        n_classes=8,
    ),
)

optimizer = dict(
    name='Adam',
    params=dict(
        lr=5e-4,
    ),
)

scheduler = dict(
    name='ReduceLROnPlateau',
    params=dict(
        factor=0.75,
        patience=2,
    ),
)

data = dict(
    train = dict(
        mode = 'train',
        dataframe = train_csv, 
        img_dir = train_images,
        loader = dict(
            shuffle=True,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=False,
        )
    ),
    valid = dict(
        mode = 'valid',
        dataframe = train_csv, 
        img_dir = train_images,
        loader = dict(
            shuffle=False,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=False,
        )
    ),
    test = dict(
        mode = 'test',
        dataframe = test_csv, 
        img_dir = test_images,
        loader = dict(
            shuffle=False,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=False,
        )
    )
)