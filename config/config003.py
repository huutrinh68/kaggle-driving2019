workdir  = 'outputs/model003'
seed = 2050
apex = False

n_fold = 15
epoch =300
resume_from = None

batch_size = 8
num_workers = 20
# imgsize = (320, 640) #(height, width)
early_stop = 10
switch_loss_epoch = 5

apex = False
gpu=[0, 1, 2, 3]

# path
train_csv = 'data/train.csv'
test_csv = 'data/sample_submission.csv'
train_images = 'data/train_images'
test_images = 'data/test_images'

model = dict(
    name='resnet18',
    params=dict(
        n_classes=8,
    ),
)

optimizer = dict(
    name='AdamW',
    params=dict(
        lr=1e-3,
    ),
)

scheduler = dict(
    name='StepLR',
    params=dict(
        step_size=3000,
        gamma=0.1,
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
        ),
        n_grad_acc = 10,
        switch_loss_epoch = switch_loss_epoch,
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
        ),
        n_grad_acc = 10,
        switch_loss_epoch = switch_loss_epoch,
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