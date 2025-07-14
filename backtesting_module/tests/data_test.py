# backtesting_module/tests/data_test.py
from ..data_handler import DataHandler   # ← one level up (backtesting_module)

def main() -> None:
    dh = DataHandler(
        alpaca_api_key="PKCLL4TXCDLRN76OGRAB",
        alpaca_secret="ig5CGnl3c1jXEepU6VK5DPXgsV5WSOBYrIJGk70T",
        polygon_key="ejp0y0ppSQJzIX1W8qSoTIvL5ja3ctO9",
    )

    # 1 ▸ get stock bars
    bars = dh.get_stock_bars("AAPL", "2023-01-01", "2023-06-30", "5Min")
    print("Found ", len(bars), " bars")

    # 2 ▸ discover call options around-the-money expiring March 2024
    calls = dh.options_search(
        "AAPL",
        exp_from="2024-03-15",
        exp_to="2024-03-22",
        strike_min=160,
        strike_max=190,
        opt_type="call",
        as_of="2024-01-01",
        limit=100,
    )
    print("Search returned these calls: \n",calls)

    # 3 ▸ pull minute bars for the first contract
    opt_bars = dh.get_option_aggregates(calls[0], "2024-01-15", "2024-02-28")
    print(opt_bars)

    print("Aggregate data for first option in search: \n", opt_bars)
if __name__ == "__main__":
    main()
