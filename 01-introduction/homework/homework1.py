import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    return mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Q1. What version of Pandas did you install?""")
    return


@app.cell
def _(pd):
    print(pd.__version__)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Loading the data""")
    return


@app.cell
def _(pd):
    fuel_efficiency_url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv"

    df_fuel = pd.read_csv(filepath_or_buffer=fuel_efficiency_url)

    df_fuel.head()
    return (df_fuel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Q2. Records count

    How many records are in the dataset?
    """
    )
    return


@app.cell
def _(df_fuel):
    df_fuel.shape[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Q3. Fuel types

    How many fuel types are presented in the dataset?
    """
    )
    return


@app.cell
def _(df_fuel):
    n_unique_fuel_type = df_fuel['fuel_type'].nunique()
    print(f"There is a total of {n_unique_fuel_type} unique fuel types")

    df_fuel['fuel_type'].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Q5. Max fuel efficiency

    What's the maximum fuel efficiency of cars from Asia?

    """
    )
    return


@app.cell
def _(df_fuel):
    cols_with_missing_values = (
        df_fuel.isnull()
               .sum()
               .sort_values(ascending=False)
               .pipe(lambda x: x[x > 0])
    )

    print(f"There is a total of {len(cols_with_missing_values)} columns with missing values: {cols_with_missing_values.index.tolist()}")

    cols_with_missing_values
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Q5. Max fuel efficiency

    What's the maximum fuel efficiency of cars from Asia?
    """
    )
    return


@app.cell
def _(df_fuel):
    df_fuel[df_fuel['origin'] == 'Asia'] ['fuel_efficiency_mpg'].max()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Q6. Median value of horsepower

    1. Find the median value of the `horsepower` column in the dataset.
    2. Next, calculate the most frequent value of the same `horsepower` column.
    3. Use the `fillna` method to fill the missing values in the `horsepower` column with the most frequent value from the previous step.
    4. Now, calculate the median value of `horsepower` once again.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""1. Find the median value of the `horsepower` column in the dataset.""")
    return


@app.cell
def _(df_fuel):
    hp_median = df_fuel['horsepower'].median()

    hp_median
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""2. Next, calculate the most frequent value of the same `horsepower` column.""")
    return


@app.cell
def _(df_fuel):
    hp_most_frequent = df_fuel['horsepower'].mode()[0]

    hp_most_frequent
    return (hp_most_frequent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""3. Use the `fillna` method to fill the missing values in the `horsepower` column with the most frequent value from the previous step.""")
    return


@app.cell
def _(df_fuel, hp_most_frequent):
    hp_mode_imputing = df_fuel['horsepower'].fillna(hp_most_frequent)

    hp_mode_imputing.isnull().sum()
    return (hp_mode_imputing,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""4. Now, calculate the median value of `horsepower` once again.""")
    return


@app.cell
def _(hp_mode_imputing):
    hp_mode_imputing.median()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Has the median changed?

    - Yes, it's value increased after applying mode imputation. The median is now the same as the mode.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Q7. Sum of weights

    1. Select all the cars from Asia
    2. Select only columns `vehicle_weight` and `model_year`
    3. Select the first 7 values
    4. Get the underlying NumPy array. Let's call it `X`.
    5. Compute matrix-matrix multiplication between the transpose of `X` and `X`. To get the transpose, use `X.T`. Let's call the result `XTX`.
    6. Invert `XTX`.
    7. Create an array `y` with values `[1100, 1300, 800, 900, 1000, 1100, 1200]`.
    8. Multiply the inverse of `XTX` with the transpose of `X`, and then multiply the result by `y`. Call the result `w`.
    9. What's the sum of all the elements of the result?

    > **Note**: You just implemented linear regression. We'll talk about it in the next lesson.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""1. Select all the cars from Asia""")
    return


@app.cell
def _(df_fuel):
    df_asia = df_fuel[df_fuel['origin'] == 'Asia']

    df_asia
    return (df_asia,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""2. Select only columns `vehicle_weight` and `model_year`""")
    return


@app.cell
def _(df_asia):
    df_asia_subset = df_asia[['vehicle_weight', 'model_year']]

    df_asia_subset
    return (df_asia_subset,)


@app.cell
def _(mo):
    mo.md(r"""3. Select the first 7 values""")
    return


@app.cell
def _(df_asia_subset):
    df_first_seven = df_asia_subset.head(7)
    return (df_first_seven,)


@app.cell
def _(mo):
    mo.md(r"""4. Get the underlying NumPy array. Let's call it `X`.""")
    return


@app.cell
def _(df_first_seven):
    X = df_first_seven.values

    X
    return (X,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""5. Compute matrix-matrix multiplication between the transpose of `X` and `X`. To get the transpose, use `X.T`. Let's call the result `XTX`.""")
    return


@app.cell
def _(X):
    XTX = X.T @ X

    XTX
    return (XTX,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""6. Invert `XTX`.""")
    return


@app.cell
def _(XTX, np):
    XTX_inv = np.linalg.inv(XTX)

    XTX_inv
    return (XTX_inv,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""7. Create an array `y` with values `[1100, 1300, 800, 900, 1000, 1100, 1200]`.""")
    return


@app.cell
def _(np):
    y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

    y
    return (y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""8. Multiply the inverse of `XTX` with the transpose of `X`, and then multiply the result by `y`. Call the result `w`.""")
    return


@app.cell
def _(X, XTX_inv, y):
    w = XTX_inv @ X.T @ y

    w
    return (w,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""9. What's the sum of all the elements of the result?""")
    return


@app.cell
def _(w):
    sum(w)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The sum of the weights is `0.51`.""")
    return


if __name__ == "__main__":
    app.run()
