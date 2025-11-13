from typing import List

import numpy as np
import pandas as pd


class Order:
    def __init__(self, df: pd.DataFrame) -> None:
        self.order_id = df.iloc[0]["order_id"]

        self.products = df[["sku", "qty", "weight_kg"]]

    def __str__(self) -> str:
        num_products = len(self.products)
        if num_products <= 3:
            products_preview = self.products.to_string(index=False)
        else:
            first_three = self.products.iloc[:3]
            products_preview = first_three.to_string(index=False) + "\n..."
        return f"Pedido {self.order_id}: {num_products} productos\n{products_preview}"


class Episode:
    def __init__(self, df) -> None:
        self.date = df.iloc[0].date
        self.orders = [Order(group[1]) for group in df.groupby(["order_id"])]

    def __len__(self):
        return len(self.orders)

    def __str__(self) -> str:
        orders_str = "\n".join([str(order) for order in self.orders])
        return f"Episodio {self.date}. Hay {len(self)} pedidos:\n{orders_str}"


class OrdersLoader:
    def __init__(self, init_date: np.datetime64) -> None:

        # data cleaning
        df = pd.read_csv("shipping_detail_report.csv", parse_dates=["Ship Date"])
        df["date"] = df["Ship Date"].dt.date

        df["Boxes"] = df["Boxes"].replace({"#DIV/0!": np.nan})
        df["Boxes"] = df["Boxes"].astype("float")
        df["Boxes"] = df["Boxes"].fillna(0).astype("uint16")

        df = df[~df["Weight [Kg]"].str.contains(",")]  # TODO: preguntar por estas filas
        df["Weight [Kg]"] = df["Weight [Kg]"].astype("float")

        df["Qty Shipped"] = df["Qty Shipped"].str.replace(",", "").astype("uint16")

        df["MOQ"] = df["MOQ"].fillna(0).astype("uint16")

        df.rename(
            columns={
                "Purchase order": "order_id",
                "ShipType": "ship_type",
                "Ship Date": "datetime",
                "SKU": "sku",
                "Qty Shipped": "qty",
                "Weight [Kg]": "weight_kg",
                "Customer": "customer",
                "MOQ": "moq",
                "Boxes": "boxes",
            },
            inplace=True,
        )

        self.episodes = {
            np.datetime64(group[0][0]): Episode(group[1])
            for group in df.groupby(["date"])
        }
        self.date = min(self.episodes.keys(), key=lambda date: np.abs(init_date - date))

    def next_days(self, n) -> List[Episode]:
        """
        De la base de datos de pedidos, samplea n epiosodios (días) adyacentes.
        """
        results = []
        keys = list(self.episodes.keys())
        i = keys.index(self.date)
        while n >= 0:
            results.append(self.episodes[keys[i]])
            i += 1
            n -= 1
        return results


if __name__ == "__main__":
    loader = OrdersLoader(np.datetime64("2025-01-30"))
    episodes = loader.next_days(2)
    print(episodes)
    # print([episode.date for episode in episodes])
    print(loader.date)
