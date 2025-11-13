import numpy as np
import pandas as pd

from orders import OrdersLoader


class WarehouseEnvCalc:
    def __init__(self, days: int, init_date: np.datetime64):

        super().__init__()

        self.orders_loader = OrdersLoader(init_date)

        self.days = days
        self.init_date = init_date
        self.current_date = init_date

        # self.shipping = self.loadDataframe("data/raw/ShippingActivityAbc.csv")
        # self.labor = self.loadDataframe("data/raw/LaborActivityAbc.csv")

    # def CalcAll(self):
    #     print(
    #         f"Calculando todos los puntajes para {self.days} dias empezando en la fecha {self.current_date}..."
    #     )
    #     print("Un mejor numero indica una mejor prediccion")
    #     ShippingABC = self.CalcDataframeScore(self.shipping, "ABC_class")
    #     print(f"Puntaje para Shipping Activity usando ABC {ShippingABC}.")
    #     ShippingAHP = self.CalcDataframeScore(self.shipping, "AHP_class")
    #     print(f"Puntaje para Shipping Activity usando AHP {ShippingAHP}.")
    #     LaborABC = self.CalcDataframeScore(self.labor, "ABC_class")
    #     print(f"Puntaje para Labor Activity usando ABC {LaborABC}.")
    #     LaborAHP = self.CalcDataframeScore(self.labor, "AHP_class")
    #     print(f"Puntaje para Labor Activity usando AHP {LaborAHP}.")

    def CalcDataframeScore(self, df: pd.DataFrame, categoryType: str) -> float:
        self.current_date = self.init_date
        days = self.orders_loader.next_days(self.days)

        reward = 0

        for day in days:
            for order in day.orders:
                products = order.products

                for sku in products["sku"]:
                    try:
                        category = (df.loc[sku])[categoryType]
                    except KeyError:
                        category = "C"  # Si no existe el valor, asumimos lo peor

                    reward += self.CalcScore(category)

        return reward

    def CalcScore(self, category) -> int:
        # Siguiendo el principio de Pareto, es mejor haber movido un objeto de categoria A que uno de categoria C
        # Por lo que se sigue la siguente proporcion
        # A -> 1, B -> 3, C -> 10

        if category == "A":
            return 1
        if category == "B":
            return 3
        return 10

    # def loadDataframe(self, df: str):
    #     df = pd.read_csv(df)
    #     df = df[["SKU", "ABC_class", "AHP_class"]]
    #     df = df.set_index("SKU")
    #     return df


# if __name__ == "__main__":
#     env = WarehouseEnvCalc(30, np.datetime64("2025-01-30"))
#     env.CalcAll()
