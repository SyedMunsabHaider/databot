# DataBot: ask plain-English questions about any table

DataBot reads any CSV or Excel file and answers questions like a junior analyst would: a direct answer first, a small table when it helps, and a chart **only if you ask for one**.

It runs on your own computer. No account, no API key and no internet are needed for normal use.

## Example

```text
>>> which product made us the most money?
Laptop has the highest total revenue: 1,094,446 (42% of the total 2,596,609). Next is Smartphone (871,440), 26% behind.

     product   revenue share %
      Laptop 1,094,446   42.1%
  Smartphone   871,440   33.6%
        Desk   184,100    7.1%

>>> revenue of Laptop in Karachi in 2025
Total revenue (product = Laptop; region = Karachi; date: 2025): 123,158
Based on 41 records. That is 4.7% of the overall total (2,596,609).

>>> compare Lahore and Karachi revenue
Total revenue (region in Lahore, Karachi): Lahore = 728,212, Karachi = 677,484. Lahore is 7.5% higher.
```

The same file works on completely different data (HR records, student results, restaurant orders...) because it reads your column names instead of assuming a fixed layout.

## What it can do

- Totals, averages, medians, counts, percentages, min / max
- Breakdowns: `average salary by department`, `revenue by month`
- Rankings: `top 5 products by profit`, `who has the highest salary?`
- Filters: `Laptop in Karachi in 2025`, `gpa above 3.5`, `last 3 months`, `salary above average`
- Comparisons: `compare Lahore and Karachi revenue`
- Relationships: `does attendance affect gpa?`, `correlation between age and salary`
- Trends over time, distributions, outliers, missing-value checks
- Follow-up questions: `what about Multan?`
- Charts only on request: add the word `chart`, e.g. `revenue by region chart`

## Quick start

```bash
pip install -r requirements.txt
python databot.py sales_data.csv
```

Or ask one question and exit:

```bash
python databot.py hr_employees.csv --ask "average salary by department"
```

In Jupyter, open `databot_demo.ipynb`, or:

```python
from databot import DataBot

bot = DataBot("your_file.csv")      # .csv or .xlsx
bot.say("which region has the highest revenue?")
```

## How it works

1. **Profiles the table:** which columns are numbers, dates, categories or names/ids (it also cleans values such as `Rs. 1,200`, `15%` and `dd/mm/yyyy` dates).
2. **Turns your question into a small query plan** (what to calculate, which groups, which filters) using built-in language rules. This is instant and free.
3. **Runs the plan with pandas.** All the maths is done by pandas on your machine; there is no AI-generated code.
4. **Optional AI backup:** if the rules cannot understand a question, a free local model ([Ollama](https://ollama.com)) or an Anthropic API key can write the same kind of plan. The AI only sees column names and a few example values, never your rows.

```python
bot.use_ollama()   # optional free local AI backup
```

## Limits

- It is rule-based, so very unusual wording can be misread. When it ignores words, it tells you.
- "Last year" and "this month" are measured from the latest date in the data, not today's date.
- It answers questions about one table at a time.
- Sample files (`sales_data.csv`, `hr_employees.csv`, `student_results.csv`) are synthetic, generated for testing.

## Files

| File | Purpose |
|---|---|
| `databot.py` | The whole bot (single file) |
| `databot_demo.ipynb` | Demo notebook |
| `sales_data.csv`, `hr_employees.csv`, `student_results.csv` | Synthetic sample datasets |
| `requirements.txt` | Python packages needed |

## Author

Munsab Haider, BS Business Analytics, FAST School of Management (NUCES), Lahore.
