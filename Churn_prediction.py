threshold = 0.8 * len(df)
Drop_cols = []
for col in df.columns:
    missing_count = df[col].isnull().sum()
    if missing_count > threshold:
        Drop_cols.append(col)
df_cleaned = df.drop(columns=Drop_cols)


numeric_columns = df_cleaned.select_dtypes(include=['number']).columns

for col in numeric_columns:
    median_value = df_cleaned[col].median()
    df_cleaned[col] = df_cleaned[col].fillna(median_value)


categorical_columns = df_cleaned.select_dtypes(include=['object']).columns


for col in categorical_columns:
    most_frequent_value = df_cleaned[col].mode()[0]
    df_cleaned[col] = df_cleaned[col].fillna(most_frequent_value)


def predict_churn(df):
    active_columns = ['Active_Days_Jan24', 'Active_Days_Feb24', 'Active_Days_mar24']

    df['Total Active Days'] = df[active_columns].sum(axis=1)

    def classify_activity(total_days):
      if total_days > 20:
        return "Active"
      else:
        return "Non-Active"
    df['Churn Prediction'] = df['Total Active Days'].apply(classify_activity)

    return df
df_cleaned = predict_churn(df_cleaned)


X = df_cleaned[['Active_Days_Jan24', 'Active_Days_Feb24', 'Active_Days_mar24']]
y = df_cleaned['Churn Prediction']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)


models = {
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "Logistic Regression": LogisticRegression()
}


best_model = None
best_accuracy = 0
accuracies = []
model_names = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy}")

    accuracies.append(accuracy)
    model_names.append(name)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

plt.figure(figsize=(8, 5))
plt.bar(model_names, accuracies, color='skyblue', edgecolor='black')
plt.xlabel('Model Name')
plt.ylabel('Accuracy Score')
plt.title('Model Accuracy Comparison')
plt.ylim(0,2)
plt.xticks(rotation=45)
plt.show()


cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)

best_model_name = best_model.__class__.__name__
average_cv_accuracy = cv_scores.mean()

print(f"Best Model: {best_model_name}")
print(f"Average Cross-Validation Accuracy: {average_cv_accuracy:.4f}")


from sklearn.metrics import classification_report

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f"Best Model: {best_model.__class__.__name__}")
print(f"Mean Absolute Error: {mae}")

joblib.dump(model, 'trained_model.pkl')

loaded_model = joblib.load('trained_model.pkl')


predictions_scores = pd.DataFrame({'Predicted_Score': model.predict(X_test)}, index=range(len(model.predict(X_test))))


predictions_scores['Churn Prediction'] = predictions_scores['Predicted_Score'].map({0: 'Active', 1: 'Non-Active'})

print(predictions_scores.head())
