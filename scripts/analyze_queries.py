import pandas as pd

train_df = pd.read_csv('data/student_query_train_5000.csv')
test_df = pd.read_csv('data/student_query_test_1000.csv')

print(f"Total unique queries in train: {train_df['Student_Query'].nunique()}")
print(f"Total unique queries in test: {test_df['Student_Query'].nunique()}")

train_queries = set(train_df['Student_Query'].unique())
test_queries = set(test_df['Student_Query'].unique())
print(f"Queries in test but not train: {test_queries - train_queries}")

print("\nAll Unique Queries in Train with their Priority and counts:")
grouped = train_df.groupby(['Student_Query', 'Priority']).size().reset_index(name='count')
for idx, row in grouped.iterrows():
    print(f"- \"{row['Student_Query']}\" -> Priority: {row['Priority']} (n={row['count']})")

