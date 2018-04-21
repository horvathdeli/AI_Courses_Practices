import tensorflow as tf
import os

_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}

def build_model_columns():
    "Builds a set of wide and deep feature columns."
    # Continuous Columns
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')

    # Categorical Column
    education = tf.feature_column.categorical_column_with_vocabulary_list(
            'education', [
                    'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
                    'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
                    '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
            'marital_status', [
                    'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
                    'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
            'relationship', [
                    'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
                    'Other-relative'])

    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
            'workclass', [
                    'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
                    'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

    # An example of hashing:
    occupation = tf.feature_column.categorical_column_with_hash_bucket(
            'occupation', hash_bucket_size=1000)
    
    # Transformations.
    age_buckets = tf.feature_column.bucketized_column(
            age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    # Wide columns 
    base_columns = [
            education, marital_status, relationship,
            workclass, occupation, age_buckets]
    
    crossed_columns = [
            tf.feature_column.crossed_column(
                    ['education', 'occupation'], hash_bucket_size=1000),
            tf.feature_column.crossed_column(
                    [age_buckets, 'education', 'occupation'], hash_bucket_size=1000)]

    wide_columns = base_columns + crossed_columns

    # Deep Columns
    deep_columns = [
            age,
            education_num,
            capital_gain,
            capital_loss,
            hours_per_week,
            tf.feature_column.indicator_column(workclass),
            tf.feature_column.indicator_column(education),
            tf.feature_column.indicator_column(marital_status),
            tf.feature_column.indicator_column(relationship),
            # An example of embedding
            tf.feature_column.embedding_column(occupation, dimension=8)]

    return wide_columns, deep_columns

def build_estimator(model_dir):
    wide_columns, deep_columns = build_model_columns()
    hidden_units = [100, 75, 50, 25]
    
    # Create the tf.estimator.RunConfig
    # to ensure that the model run on CPU
    # because in this case it is faster
    run_config = tf.estimator.RunConfig().replace(
            session_config = tf.ConfigProto(device_count = {'GPU': 0}))
    
    return tf.estimator.DNNLinearCombinedClassifier(
            model_dir = model_dir,
            linear_feature_columns = wide_columns,
            dnn_feature_columns = deep_columns,
            dnn_hidden_units = hidden_units,
            config = run_config)
    
def input_fn(data_file, num_epochs, shuffle, batch_size):
    "Generate an input function for the Estimator."       
    def parse_csv(value):
        print("Parsing", data_file)
        columns = tf.decode_csv(value,
                                record_defaults = _CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('income_bracket')
        return features, tf.equal(labels, '>50K')
    
    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size = _NUM_EXAMPLES['train'])
    
    dataset = dataset.map(parse_csv, num_parallel_calls = 5)
    
    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset

def train_input_fn():
    return input_fn(train_file, epochs_between_evals, True, batch_size)

def eval_input_fn():
    return input_fn(test_file, 1, False, batch_size)

# HyperParameters
train_epochs = 40
epochs_between_evals = 2
batch_size = 40

# Build the estimator
model_dir="C:/Users/u20f16/Documents/GitHub/Library-for-builded-models-by-me/Temp/TensorFlowTutorial/model"
model = build_estimator(model_dir)

# Load the data
data_dir = "C:/Users/u20f16/Documents/GitHub/Library-for-builded-models-by-me/Temp/TensorFlowTutorial/data"
train_file= os.path.join(data_dir, "adult.data")
test_file = os.path.join(data_dir, "adult.test")

# Train and evaluate the model every `flags.epochs_between_evals` epochs.
for n in range (train_epochs // epochs_between_evals):
    model.train(input_fn = train_input_fn)
    results = model.evaluate(input_fn = eval_input_fn)

# Display evaluation metrics
print('Results at epoch', (n+1) * epochs_between_evals)
print('-' * 100)



