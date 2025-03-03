import marimo

__generated_with = "0.11.12"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        Wait a sec, the code is running.
            _(it runs in WebAssembly, isn't it cool? Python, numpy, sklearn. On your laptop in a browser! Even on your phone, allthough the support is poor for now)_
        """
    )
    return


@app.cell
def _(
    intercept_err,
    lines,
    mo,
    n_experiments,
    n_points,
    rep_error_pred,
    repr_error_by_n_points,
    scatter,
    slope_err,
    true_slope,
):
    mo.md(
        f"""
        # Experiment

        /// admonition | Disclaimer

        I'm pretty sure we were just discussing the old plain overfitting on test data, I just struggled to properly phrase it. I decided to create a cool visualization nevertheless :)
        ///

        > I recommend pressing `CMD/CTRL+.` (CMD + dot) to toggle app view (hide code).

        Please, select the number of points: {n_points}, and slope (just for fun): {true_slope}

        1. We generate **{n_points.value}** data points: x values with constant step, and y values (with and without noise):
        {mo.as_html(scatter.configure_scale(continuousPadding=1.0))}

        2. We know the true line, and we can fit the linear regression to find the predicted line:
        {mo.as_html((scatter + lines).configure_scale(continuousPadding=1.0))}

        3. We compute the RMSE for the noisy points with the model (predicted line) (Here, I use the RMSE instead of reprojection error as we are not dealing with computer vision):

        - **RMSE for predicted line: {rep_error_pred:.2f}**
        - **Slope error: {slope_err:.2f}**
        - **Intercept error: {intercept_err:.2f}**

        # Simulate multiple experiments

        To illustrate that, let's calculate the RMSE for the predicted line for different number of points multiple times and aggregate the RMSEs. Please, select the number of experiments: {n_experiments}
        {mo.as_html(repr_error_by_n_points)}

        My findings are the following:

        1. The more points, the closer the model to the ground truth.
        2. The more points, the higher the RMSE with the predicted line (approaching the real noise data).

        Well, from that I conclude that the RMSE for 20 points would be higher then for 10 points (if we calculate against the predicted model. We typically don't know the ground truth (for example, the ground truth camera parameters)).
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import polars as pl
    from sklearn.linear_model import LinearRegression
    import altair as alt
    import marimo as mo
    return LinearRegression, alt, mo, np, pl


@app.cell
def _(np):
    np.random.seed(42)
    return


@app.cell
def _(LinearRegression, mo, np):
    def generate_noisy_points(slope, intercept, n_points, noise_std):
        x = np.linspace(0, 10, n_points)
        y_true = slope * x + intercept
        y_noisy = y_true + np.random.normal(0, noise_std, n_points)
        return x, y_noisy, y_true


    # Function to fit a line using least squares
    def fit_line(x, y):
        model = LinearRegression()
        x_reshaped = x.reshape(-1, 1)
        model.fit(x_reshaped, y)
        return model.coef_[0], model.intercept_


    def compute_rmse(x, y, slope, intercept):
        y_pred = slope * x + intercept
        error = np.sqrt(np.sum((y - y_pred) ** 2) / len(x))
        # error = np.sum(np.abs((y - y_pred))) / len(x)
        return error


    # Parameters
    # true_slope = 2
    true_slope = mo.ui.slider(-10, 10, 0.1, 2)
    true_intercept = 1
    n_points = mo.ui.slider(2, 50, 1)
    noise_std = 5.0
    return (
        compute_rmse,
        fit_line,
        generate_noisy_points,
        n_points,
        noise_std,
        true_intercept,
        true_slope,
    )


@app.cell
def _(
    fit_line,
    generate_noisy_points,
    n_points,
    noise_std,
    true_intercept,
    true_slope,
):
    x, y_noisy, y_true = generate_noisy_points(true_slope.value, true_intercept, n_points.value, noise_std)
    slope, intercept = fit_line(x, y_noisy)
    _, _, y_pred = generate_noisy_points(slope, intercept, n_points.value, noise_std)
    return intercept, slope, x, y_noisy, y_pred, y_true


@app.cell
def _(pl, x, y_noisy, y_pred, y_true):
    _df = pl.DataFrame({"x": x, "y_noisy": y_noisy, "y_true": y_true, "y_pred": y_pred}).unpivot(index=["x"])

    scatter = _df.filter(pl.col("variable") != "y_pred").plot.scatter("x", "value", "variable")
    lines = _df.filter(pl.col("variable") != "y_noisy").plot.line("x", "value", color="variable")
    return lines, scatter


@app.cell
def _(
    compute_rmse,
    intercept,
    np,
    slope,
    true_intercept,
    true_slope,
    x,
    y_noisy,
):
    rep_error_pred = compute_rmse(x, y_noisy, slope, intercept)
    slope_err, intercept_err = np.abs(true_slope.value - slope), np.abs(true_intercept - intercept)
    return intercept_err, rep_error_pred, slope_err


@app.cell
def _(mo):
    n_experiments = mo.ui.slider(1, 500, 1, 200)
    return (n_experiments,)


@app.cell
def _(
    compute_rmse,
    fit_line,
    generate_noisy_points,
    mo,
    n_experiments,
    noise_std,
    np,
    true_intercept,
    true_slope,
):
    data = []

    for n in mo.status.progress_bar(range(2, 50), title="Running experiments"):
        for _ in range(n_experiments.value):
            _x, _y_noisy, _y_true = generate_noisy_points(true_slope.value, true_intercept, n, noise_std)
            _slope, _intercept = fit_line(_x, _y_noisy)
            _rmse = compute_rmse(_x, _y_noisy, _slope, _intercept)
            _slope_err, _intercept_err = np.abs(true_slope.value - _slope), np.abs(true_intercept - _intercept)
            data.append(
                {
                    "n_points": n,
                    "rmse": _rmse,
                    "slope_err": _slope_err,
                    "intercept_err": _intercept_err,
                }
            )
    return data, n


@app.cell
def _(data, n_experiments, pl):
    df = pl.DataFrame(data)
    df = (
        df.group_by("n_points")
        .agg(pl.col("rmse").mean(), pl.col("slope_err").mean(), pl.col("intercept_err").mean())
        .unpivot(index=["n_points"])
        .with_columns(
            pl.when(pl.col("variable") == "rmse")
            .then(pl.lit("RMSE error"))
            .otherwise(pl.lit("Model error"))
            .alias("Error type")
        )
    )
    repr_error_by_n_points = df.plot.scatter("n_points", "value", "variable", column="Error type").properties(
        title=f"Aggregated errors for {n_experiments.value} experiments"
    )
    return df, repr_error_by_n_points


if __name__ == "__main__":
    app.run()
