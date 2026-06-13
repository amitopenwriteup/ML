A **Random Forest** is a machine learning algorithm that combines many **decision trees** and lets them vote on the final answer.

Think of it like asking 100 people for their opinion instead of relying on just one person.

---

## Step 1: What is a Decision Tree?

A decision tree makes predictions by asking a series of questions.

For example, to predict whether a customer will buy a product:

```text
Age > 30?
├── Yes
│   ├── Salary > 50,000?
│   │   ├── Yes → Buy
│   │   └── No  → Don't Buy
│
└── No
    └── Don't Buy
```

The tree keeps splitting the data based on conditions until it reaches a prediction.

---

## Step 2: Why One Tree Isn't Enough

A single decision tree can easily **overfit**.

Imagine learning from only a few examples:

```text
Tree says:
"If age is 31, always buy."
```

That rule might work for training data but fail on new data.

---

## Step 3: Random Forest Creates Many Trees

Instead of one tree:

```text
Tree 1
Tree 2
Tree 3
...
Tree 100
```

Each tree is trained on a slightly different random subset of the data.

That's where the **"Random"** in Random Forest comes from.

---

## Step 4: Voting

Suppose we want to predict whether a customer will buy:

```text
Tree 1 → Yes
Tree 2 → Yes
Tree 3 → No
Tree 4 → Yes
Tree 5 → Yes
```

Result:

```text
Yes wins 4 votes to 1
```

Final prediction:

```text
Buy = Yes
```

This is called **majority voting**.

---

## For Regression Problems

If predicting a number instead of a category:

Example:

```text
Tree 1 → 100
Tree 2 → 120
Tree 3 → 110
```

Average:

```text
(100 + 120 + 110) / 3 = 110
```

Final prediction:

```text
110
```

---

## Why It Works Well

A single tree may make mistakes.

```text
Tree 1 → Wrong
Tree 2 → Wrong
Tree 3 → Correct
...
```

When many trees vote, individual mistakes tend to cancel out.

This usually gives:

* Better accuracy
* Less overfitting
* More stable predictions

---

## Parameters in Your Code

You used:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
```

### n_estimators

```python
n_estimators=100
```

Means:

```text
Build 100 trees
```

More trees generally improve stability but take longer to train.

---

### max_depth

```python
max_depth=5
```

Limits how deep each tree can grow.

```text
Depth 1 → 1 question
Depth 5 → up to 5 levels of questions
```

Small depth:

* Simpler model
* Less overfitting

Large depth:

* More complex model
* Can overfit

---

### random_state

```python
random_state=42
```

Ensures you get the same random choices every time you run the code.

---

## Simple Example

Suppose we're predicting whether a student passes an exam.

Features:

```text
Hours Studied
Attendance
Assignments Submitted
```

Input:

```text
Hours Studied = 8
Attendance = 90%
Assignments = Yes
```

Trees vote:

```text
Tree 1 → Pass
Tree 2 → Pass
Tree 3 → Fail
Tree 4 → Pass
Tree 5 → Pass
```

Result:

```text
Pass (4 votes)
```

Think of your code as **teaching a student and then giving an exam**.

---

## Step 1: Read the data

```python
df = pd.read_csv("data/raw/dataset.csv")
```

This loads the CSV file into memory.

Example:

| age | income | credit_score | target |
| --- | ------ | ------------ | ------ |
| 25  | 30000  | 700          | 1      |
| 40  | 50000  | 600          | 0      |

The data becomes a table called `df`.

---

## Step 2: Separate Questions and Answers

```python
X = df.drop("target", axis=1)
y = df["target"]
```

Imagine a teacher giving:

### Questions (X)

```text
age
income
credit_score
loan_amount
...
```

### Answers (y)

```text
target
```

Example:

| Age | Income | Target |
| --- | ------ | ------ |
| 25  | 30000  | 1      |
| 40  | 50000  | 0      |

The model's job is:

```text
Given Age + Income + Credit Score
Predict Target
```

---

## Step 3: Split Data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

Suppose you have 1000 rows.

```text
800 rows → Training
200 rows → Testing
```

Training data teaches the model.

Testing data checks whether it learned correctly.

---

## Step 4: Create the Model

```python
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
```

### Random Forest = 100 decision trees

Imagine asking 100 experts:

```text
Expert 1 → Yes
Expert 2 → No
Expert 3 → Yes
...
```

The majority answer wins.

That's exactly what Random Forest does.

```python
n_estimators=100
```

means:

```text
Create 100 trees
```

```python
max_depth=5
```

means:

```text
Each tree can ask at most 5 levels of questions
```

---

## Step 5: Train the Model

```python
model.fit(X_train, y_train)
```

This is the learning step.

The model studies examples like:

```text
Age=25 Income=30000 → Target=1
Age=40 Income=50000 → Target=0
Age=35 Income=45000 → Target=1
```

and learns patterns.

---

## Step 6: Take the Exam

```python
preds = model.predict(X_test)
```

Now the model sees new rows it never saw before.

Example:

```text
Age=30 Income=40000
```

and predicts:

```text
Target = 1
```

---

## Step 7: Check Marks

### Accuracy

```python
accuracy_score(y_test, preds)
```

Measures:

```text
How many answers were correct?
```

Example:

```text
200 questions
160 correct
```

Accuracy:

```text
160 / 200 = 80%
```

---

### F1 Score

```python
f1_score(y_test, preds)
```

Another way to measure quality.

Range:

```text
0 = bad
1 = perfect
```

---

## Step 8: Save the Trained Model

```python
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)
```

After learning, save the model to a file:

```text
models/model.pkl
```

Think of it as:

```text
Train once
Save the brain
Reuse later
```

Without saving:

```text
Every time you start
you must train again
```

With saving:

```text
Load the trained model
and predict immediately
```

---

## Complete Flow

```text
dataset.csv
      ↓
Load Data
      ↓
Separate Features & Target
      ↓
Split Train/Test
      ↓
Train Random Forest
      ↓
Predict on Test Data
      ↓
Calculate Accuracy & F1
      ↓
Save model.pkl
```

In one sentence:

**This script teaches a Random Forest model using historical data, tests how well it learned, and saves the trained model for future use.**


---

## Why Random Forest Is Popular

✅ Easy to use
✅ Works well on many datasets
✅ Handles missing patterns well
✅ Usually needs little tuning
✅ Good baseline model for classification and regression

Because of these advantages, Random Forest is often one of the first models data scientists try before moving to more advanced methods like XGBoost, LightGBM, or neural networks.
