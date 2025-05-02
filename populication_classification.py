import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Set style for visualizations
plt.style.use('seaborn')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# Load the data
print("Loading and preprocessing data...")
df = pd.read_csv('Original data\Population estimates by gender nationality and region 2010 - 2022_data.csv')

# Data cleaning
df['Year'] = df['Year'].str.replace(',', '').astype(int)
df['Population estimates'] = df['Population estimates'].str.replace(',', '').astype(float)

# Create bins for population estimates to make it a classification problem
bins = [0, 100000, 500000, 1000000, float('inf')]
labels = ['Small', 'Medium', 'Large', 'Very Large']
df['Population Category'] = pd.cut(df['Population estimates'], bins=bins, labels=labels)

# Plot population distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Population Category', data=df)
plt.title('Distribution of Population Categories')
plt.savefig('Visualizations/population_distribution.png')
plt.close()

# Encode categorical variables
label_encoders = {}
for col in ['Region', 'Gender', 'Nationality', 'Population Category']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Select features and target
X = df[['Region', 'Year', 'Gender', 'Nationality']]
y = df['Population Category']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Plot train-test distribution
plt.figure(figsize=(10, 6))
sns.countplot(x=label_encoders['Population Category'].inverse_transform(y_train))
plt.title('Training Set Distribution')
plt.savefig('Visualizations/train_distribution.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.countplot(x=label_encoders['Population Category'].inverse_transform(y_test))
plt.title('Test Set Distribution')
plt.savefig('Visualizations/test_distribution.png')
plt.close()

# Scale numerical features
scaler = StandardScaler()
X_train[['Year']] = scaler.fit_transform(X_train[['Year']])
X_test[['Year']] = scaler.transform(X_test[['Year']])

# Save preprocessed data 
X_train.to_csv('Preprocessed data\X.csv', index=False)
X_test.to_csv('Preprocessed data\X_test.csv', index=False)
y_train.to_csv('Preprocessed data\Y.csv', index=False, header=True)
y_test.to_csv('Preprocessed data\Y_test.csv', index=False, header=True)
print("Preprocessing complete. Saved files: X.csv, X_test.csv, Y.csv, Y_test.csv")

# Define models
models = {
    'SVM': SVC(kernel='rbf', C=10, gamma=0.1, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'NaiveBayes': GaussianNB()
}

# Train and evaluate models
results = []
print("\nTraining models...")

for name, model in models.items():
    print(f"\nTraining {name}...")
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoders['Population Category'].classes_, zero_division=0)
    results.append({'Model': name, 'Accuracy': accuracy})
    
    # Save predictions with original category names
    pred_df = pd.DataFrame({
        'Actual': label_encoders['Population Category'].inverse_transform(y_test),
        'Predicted': label_encoders['Population Category'].inverse_transform(y_pred)
    })
    pred_df.to_csv(f'Result\prediction_{name}_model.csv', index=False)
    
    print(f"{name} Classification Report:")
    print(report)

# Create results dataframe
results_df = pd.DataFrame(results)

# Plot model accuracy comparison
plt.figure(figsize=(12, 6))
ax = sns.barplot(x='Model', y='Accuracy', data=results_df)
plt.title('Model Accuracy Comparison', fontsize=20)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.xlabel('Model', fontsize=14)

# Add accuracy values on top of bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}%', 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', 
                xytext=(0, 10), 
                textcoords='offset points',
                fontsize=12)

plt.savefig('Visualizations/model_accuracy_comparison.png', bbox_inches='tight', dpi=300)
plt.close()


# Print results summary
print("\nModel Performance Summary:")
print(results_df.to_string(index=False))
print("\n\nAll operations completed successfully!")

