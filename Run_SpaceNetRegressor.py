import numpy as np
import pandas as pd
import os
import time
import logging
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from nilearn.decoding import SpaceNetClassifier, SpaceNetRegressor
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_roi
import nilearn.image as image
from nilearn.image import load_img
from sklearn.svm import SVC
from sklearn.externals import joblib
from nilearn.image import new_img_like
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

# SETUP LOGGING
# Get timestamp as string
timestamp = time.strftime('%Y%m%d-%H%M%S')
taskcode='SMA'
predictor='PPVT'
language='en' # en or sp

start = datetime.now()
logdir = './outputs/%s-%sx%s/' % (timestamp,taskcode,predictor) # directory
logname = './outputs/%s-%sx%s/%s_SpaceNet_log.txt' % (timestamp,taskcode,predictor,timestamp)
if not os.path.exists(logdir):
    os.makedirs(logdir)  # create if missing
logging.basicConfig(
    filename=logname,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s : %(message)s',
    datefmt='%d %b %Y %H:%M:%S')

# 0. ANALYSIS PARAMETERS
inputcsv = 'mvpa_meta_ENvsSP_sma.csv'  # csv with participant info & file paths
fitpenalty = 'tv-l1'  # regularisation using tv-l1 or graph-net
screenp = 20  # screening percentile, default 20
mnimask = './MNI_brain_mask.nii.gz' # mask for data during fit

logging.info('Timestamp: ' + timestamp)
logging.info('PARAMS - ' + str(taskcode)+'x'+str(predictor) + ' lang: ' + str(language) + ' fit: ' + str(fitpenalty) + ', screen: ' + str(screenp))

######################################################################

# 1. LOAD THE DATA
# Load csv
rawdata = pd.read_csv(inputcsv)

inputdata = rawdata.query("lang == @language")

# Get array X of .nii file paths from the "image" column
X = np.array(inputdata['image'])
# Get array Y
y = np.array(inputdata['ppvtraw'])  # 'phonemes' 'ppvtraw' etc

logging.info('Data loaded.')

# Sort data for better visualization
perm = np.argsort(y)
y = y[perm]
X = X[perm]

# 2. CONSTRUCT TRAINING & TEST DATA
rng = check_random_state(42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=.8, random_state=rng)
###########################################################################
# Fit the SpaceNet and predict with it
decoder = SpaceNetRegressor(memory="nilearn_cache", penalty=fitpenalty,
                            screening_percentile=screenp, memory_level=2, n_jobs=4)

logging.info('Classifier set up. Starting model fit.')
decoder.fit(X_train, y_train)  # fit
logging.info('Model fit complete. Saving outputs...')

y_pred = decoder.predict(X_test).ravel()  # predict
mse = np.mean(np.abs(y_test - y_pred))
logging.info('Mean square error (MSE) of the prediction: %.2f' % mse)

# 4. OUTPUTS
# Save csv metadata
inputdata.to_csv(logdir+inputcsv)
# Save decoder object
decoder_dir = './outputs/%s-%sx%s/decoder_sma/' % (timestamp,taskcode,predictor) # directory
if not os.path.exists(decoder_dir):
    os.makedirs(decoder_dir)  # create if missing
dec_filename = '%s%s_decoder_SpaceNet_%sscreen%d.jbl' % (
    decoder_dir, timestamp, decoder.penalty, screenp)
joblib.dump(decoder, dec_filename)

# Save coefficients to nifti file
coef_dir = './outputs/%s-%sx%s/coefs_sma/' % (timestamp,taskcode,predictor) # directory
if not os.path.exists(coef_dir):
    os.makedirs(coef_dir)  # create if missing
coef_filename = '%s%s_coefs_SpaceNet_%sscreen%d.nii' % (
    coef_dir, timestamp, decoder.penalty, screenp)
coef_img = decoder.coef_img_
coef_mask = load_img(mnimask)
coef_masked = new_img_like(coef_img, coef_img.get_data().squeeze() * coef_mask.get_data())
coef_masked.to_filename(coef_filename)
duration = datetime.now() - start
logging.info('Script ran for: ' + str(duration))
logging.info('All outputs saved.')

###########################################################################
# Visualize the quality of predictions
# -------------------------------------
import matplotlib.pyplot as plt
plt.figure()
plt.suptitle("Mean Absolute Error %.2f" % mse)
linewidth = 3
ax1 = plt.subplot('211')
ax1.plot(y_test, label="True value", linewidth=linewidth)
ax1.plot(y_pred, '--', c="g", label="Predicted value", linewidth=linewidth)
ax1.set_ylabel("value")
plt.legend(loc="best")
ax2 = plt.subplot("212")
ax2.plot(y_test - y_pred, label="True value - predicted value",
         linewidth=linewidth)
ax2.set_xlabel("dataset")
plt.legend(loc="best")

plt.show()
