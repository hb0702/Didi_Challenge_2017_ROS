## How to train deep leaning model for car Detection
1. Run `python convert_to_full_view_panorama.py` to create training panoramic views.

2. Run `python full_view_train.py` to train the model

    2.1. Set `continue_training = False` for first time training.

    2.2. Set `continue_training = True` for continue_training. Modify `saved_model` path for continue_training.

3. Trained models will be saved in folder `saved_model`
