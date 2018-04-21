# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 11:07:14 2018

@author: u20f16
"""

# =============================================================================
# Print
# =============================================================================
text = "Coolshit"

# These 2 are the same  :P
print("%s \n" % text)

print("{} \n".format(text))

# =============================================================================
# Understand:
# tf.feature_column.categorical_column_with_hash_bucket()
# =============================================================================
import tensorflow as tf
occupation = tf.feature_column.categorical_column_with_hash_bucket('Engineer',
                                                                   hash_bucket_size = 20)

print("\n Occupation: {}".format(occupation))

# =============================================================================
# STEPS
# =============================================================================

# 1. processing the input data and defining all 
# the feature columns

# 2. defining the model

# 3. training the model

# 4. evaluating the model






