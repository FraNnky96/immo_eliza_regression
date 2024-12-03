import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression


# Load the cleaned dataset
df = pd.read_csv('data\\clean_data.csv')
# Adding feature

# Defining target and features
X = df.drop(["Price"], axis=1)
X = pd.get_dummies(X, columns=["Property","Locality", "Region", "Property type", "Building condition", "Province"])
y = df["Price"]


# Apply SelectKBest with ANOVA F-value
selector = SelectKBest(score_func=f_regression, k=10)
selector.fit(X, y)

# Displaying the feature 10 best features_scores
top_feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': selector.scores_})
top_feature_scores = top_feature_scores.nlargest(10, 'Score')
print(top_feature_scores)

# Visualize the feature scores
plt.figure(figsize=(10, 6))
bars = plt.barh(top_feature_scores['Feature'], top_feature_scores['Score'], color='skyblue')

# Adding annotations (F-scores) to the bars
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
             f'{bar.get_width():.2f}', va='center', ha='left', color='black', fontsize=10)

plt.xlabel('F-Score')
plt.ylabel('Features')
plt.title('Feature Importance Using ANOVA F-Value')
plt.show()

