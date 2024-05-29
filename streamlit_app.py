import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt
import streamlit as st

# Function for the Markov chain generator
def generate_markov_chain(num_points, alpha, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    series = np.zeros(num_points)
    series[0] = np.random.randn()
    
    for t in range(1, num_points):
        series[t] = alpha * series[t-1] + np.random.randn()
    
    return series

# Function to apply moving average filter using convolution
def moving_average(series, window_size):
    window = np.ones(window_size) / window_size
    return np.convolve(series, window, mode='valid')

# Streamlit UI
st.title(f"Ανάλυση χρονοσειρών/διαδικασιών Markov")
st.markdown("-----")
st.write("**Επιλογές**")
num_points = st.slider("Μέγεθος χρονοσειράς", min_value=10, max_value=1000, value=100)
window_size = st.slider("Μέγεθος παραθύρου (Moving Average)", min_value=1, max_value=50, value=5)
num_lags = st.slider("Lags (auto-correlation). Προτείνεται να είναι περίπου **$\sqrt[]{Μέγεθος χρονοσειράς}$**", min_value=1, max_value=35, value=10)
random_seed = st.number_input("Random Seed", value=42, step=1)
a_values = [0, 0.2, 0.5, 0.8, 1.0, -0.5, -0.8, 1.2]
st.markdown("-----")

# Generate series for each coefficient
series_dict = {f"α = {a}": generate_markov_chain(num_points, a, random_seed) for a in a_values}

# Dictionary of DataFrames
dfs = {}
for key, series in series_dict.items():
    df = pd.DataFrame({
        "Value": series
    })
    df.index.name = "Index"
    dfs[key] = df

# Dictionary to store analysis results
analysis_results = {}
for key, series in series_dict.items():
    results = {}
    results["mean_value"] = np.mean(series)
    results["variance_value"] = np.var(series)
    results["auto_corr"] = sm.tsa.acf(series, nlags=num_lags)
    results["partial_auto_corr"] = sm.tsa.pacf(series, nlags=num_lags)
    adf_test = sm.tsa.adfuller(series)
    results["is_stationary"] = adf_test[1] < 0.05  # p-value from ADF test
    results["adf_p_value"] = adf_test[1]
    results["kurtosis_value"] = kurtosis(series)
    results["skewness_value"] = skew(series)
    results["ljung_box_test"] = sm.stats.acorr_ljungbox(series, lags=[num_lags], return_df=True)
    
    analysis_results[key] = results

# Plot and print the results
for key, series in series_dict.items():
    # Apply moving average filter
    smoothed_series = moving_average(series, window_size)
    
    st.header("Γραφική παράσταση χρονοσειράς με"
    # Plot the original and smoothed series(achieved with moving aberage)
    fig, ax = plt.subplots()
    ax.plot(range(num_points), series, label="Original Series")
    ax.plot(range(window_size-1, num_points), smoothed_series, label="Smoothed Series", color='orange')
    ax.set_title(f"{key}")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)
    
    # Retrieve the analysis results
    results = analysis_results[key]
    
    # Create a DataFrame for auto-correlation and partial auto-correlation
    auto_corr_df = pd.DataFrame({
        "Lag": range(num_lags + 1),
        "Auto-correlation": results['auto_corr'],
        "Partial Auto-correlation": results['partial_auto_corr']
    })
    
    # Display results
    st.write(f"**• Mean**: {results['mean_value']:.4f}")
    st.write("Η μέση τιμή των τιμών της διασποράς")

    st.write(f"**• Variance**: {results['variance_value']:.4f}")
    st.write("Η διασπορά ή variance δείχει πόσο διασκορπισμένες είναι οι τιμές της χρονοσειράς ή δείχνει την **απόκλιση** από τη μέση τιμή (mean)")

    auto_corr_df.set_index("Lag", inplace=True)
    st.write("**• Auto-correlation και Partial Auto-correlation**:")
    st.write(auto_corr_df)
    st.write("Το auto-correlation ή η αυτοσυσχέτιση μετρά τη γραμμική σχέση της χρονοσειράς σε διαφορετικούς χρόνους (lags). Υψηλό auto-correlation σημαίνει πως οι τιμές της χρονοσειράς εξαρτώνται από τις προηγούμενες τιμές της. Δείχνει δηλαδή την ύπαρξη ή μη, **μνήμης**")
    st.write("Η μερική αυτοσυσχέτιση μετρά τη γραμμική σχέση της τρέχουσας τιμής της χρονοσειράς με τις προηγούμενες τιμές της, **αφαιρώντας την επίδραση των ενδιάμεσων καθυστερήσεων(time lags)**.")
    
    # Show the timeseries DataFrame
    st.write(f"**• Τιμές χρονοσειράς**")
    st.dataframe(dfs[key])
    st.markdown("-----")


    st.header("Παρατηρήσεις")
    st.write("Φαίνεται πως όσο αυξάνεται το α, δηλαδή η μεταβλητή που επηρεάζει τη κάθε νέα τιμή της χρονοσειράς, τόσο περισσότερο αυξάνονται οι μικρές διακυμάνσεις και οι αλλαγές στις τιμές της χρονοσειράς. Ένα υψηλό α μπορεί να οδηγήσει σε μια ασταθή χρονοσειρά της οποίας οι τιμές αλλάζουν γρήγορα.  ")
    st.write("Ενδεικτικό παράδειγμα είναι η γραφική παράσταση της χρονοσειράς με α = 1.2, που οδηγεί σε μια  εκθετική κατανομή των τιμών")
