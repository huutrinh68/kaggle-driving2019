workdir  = 'model001'
seed = 2050
apex = False

n_fold = 15
epoch =300
resume_from = None

batch_size = 22
num_workers = 10
# imgsize = (320, 640) #(height, width)
early_stop = 10


# path
train_csv = 'data/train.csv'
test_csv = 'data/test.csv'
train_images = 'data/train_images'

train_mode = True

data = dict(
    train = dict(
        train_csv = train_csv, 
        loader = dict(
            shuffle=True,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=False,
        )
    ),
    valid = dict(

    ),
    test = dict(

    )
)