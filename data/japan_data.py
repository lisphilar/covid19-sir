#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import openpyxl
import pandas as pd


def main():
    # Filenames
    dir_path = Path(__file__).with_name("japan")
    excel_path = dir_path.joinpath("covid_jpn.xlsx")
    total_path = dir_path.joinpath("covid_jpn_total.csv")
    pref_path = dir_path.joinpath("covid_jpn_prefecture.csv")
    # Register new total values in the excel file
    register_total(excel_path, total_path)
    # Register new values at prefecture level in the excel file
    register_prefecture(excel_path, pref_path)


def register_total(excel_path, csv_path):
    """
    Copy the new total values in "Conv-total" sheet and paste them to "total" sheet.
    Then, update the result to the CSV file.

    Args:
        excel_path (str or pathlib.Path): excel file to read the records
        csv_path (str or pathlib.Path): CSV file to save the records
    """
    # Read values
    old_df = pd.read_excel(excel_path, sheet_name="total")
    new_df = pd.read_excel(
        excel_path, sheet_name="Conv-total", header=1, nrows=3)
    df = pd.concat([old_df, new_df], ignore_index=True)
    df = df.drop_duplicates(subset=["Date", "Location"], keep="last")
    df = df.loc[~(df["Date"].isna() | df["Location"].isna())]
    df = df.drop("Unnamed: 14", axis=1, errors="ignore")
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df = df.reset_index(drop=True)
    # Add new records to "total" sheet
    wb = openpyxl.load_workbook(excel_path)
    wb.remove(wb["total"])
    wb.save(excel_path)
    with pd.ExcelWriter(excel_path, mode="a") as fh:
        df.to_excel(fh, sheet_name="total", index=False)
    # Save as CSV file
    df.to_csv(csv_path, index=False)
    # Show the new records
    print(
        f"\n'total' sheet of {excel_path} and {csv_path} was successfully updated.")
    print(df.tail(6))


def register_prefecture(excel_path, csv_path):
    """
    Copy the new values in "Conv-prefecture" sheet and paste them to "prefecture" sheet.
    Then, update the result to the CSV file.

    Args:
        excel_path (str or pathlib.Path): excel file to read the records
        csv_path (str or pathlib.Path): CSV file to save the records
    """
    print("regirster_prefecture function is ineffective because it coverts all new values to NA values.")
    return
    # Read values
    old_df = pd.read_excel(excel_path, sheet_name="prefecture")
    new_df = pd.read_excel(
        excel_path, sheet_name="Conv-prefecture", header=3, nrows=47)
    new_df = new_df.iloc[:, 1:]
    df = pd.concat([old_df, new_df], ignore_index=True)
    df = df.drop_duplicates(subset=["Date", "Prefecture"], keep="last")
    df = df.loc[~(df["Date"].isna() | df["Prefecture"].isna())]
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df = df.reset_index(drop=True)
    # Add new records to "prefecture" sheet
    wb = openpyxl.load_workbook(excel_path)
    wb.remove(wb["prefecture"])
    wb.save(excel_path)
    with pd.ExcelWriter(excel_path, mode="a") as fh:
        df.to_excel(fh, sheet_name="prefecture", index=False)
    # Save as CSV file
    df.to_csv(csv_path, index=False)
    # Show the new records
    """
    print(
        f"\n'total' sheet of {excel_path} and {csv_path} was successfully updated.")
    print(df.tail(6))
    """


if __name__ == "__main__":
    main()
