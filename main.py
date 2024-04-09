import DataPreperation, Model_Building, Constants, Submission
if __name__ == '__main__':
    df, serie = DataPreperation.data_creation()
    DataPreperation.check_outliers(df, serie)
    X,y = DataPreperation.getInputModel()
    model,x_test, y_test = Model_Building.model_build(X,y)
    Model_Building.model_evaluate(model, x_test, y_test)
    Submission.create_submission(model)
    