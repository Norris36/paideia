import pandas as pd
import numpy as np

def set_individual_price(row, conversion):
    amount = row['amounts']
    purchase_size = row['purchase_size']
    price = row['price']
    input_units = row['units']
    purchase_unit = row['purchase_unit']

    # Early return for zero purchase size
    if purchase_size == 0:
        return 0

    # Check for unit conversion
    conversion_filter = (conversion.units == input_units) & (conversion.purchase_unit == purchase_unit)
    conversion_row = conversion[conversion_filter]
    if not conversion_row.empty:
        conversion_rate = conversion_row.rate.values[0]
    else:
        return 0

    # Calculate price based on unit matching and conversion
    if input_units == purchase_unit:
        total_price = np.ceil(amount / purchase_size) * price if amount > purchase_size else price
    elif input_units == 'g' and purchase_unit == 'stk':
        total_price = price
    else:
        total_price = np.ceil(amount * conversion_rate / purchase_size) * price if amount > purchase_size else price

    return total_price

def ingredient_divider(url, pantry):
    """This function takes a url and returns the number of ingredients in the recipe that are in the urls ingredients list.

    Args:
        url (str): the url from a recipe

    Returns:
        float: the percentage of ingredients in the recipe that are in the urls ingredients list
    """
    local_ingredient_amount = len(pantry[pantry.urls == url])
    local_matches = len(pantry[(pantry.urls == url) & (pantry.in_recipe == True)])
    if local_ingredient_amount == 0 or local_matches == 0:
        return 0
    else:
        return local_matches / local_ingredient_amount

def matching_ingredients(url, pantry, recipes):
    """This function takes a url and updates the recipes dataframe with a column that shows the percentage of ingredients in the recipe that are in the urls ingredients list.

    Args:
        url (_type_): _description_
    """
    sample_ingredients = pantry[pantry.urls == url].ingredients.tolist()
    pantry['in_recipe'] = False
    pantry.loc[pantry.ingredients.isin(sample_ingredients), 'in_recipe'] = True
    recipes['matches'] = recipes.urls.apply(lambda url : ingredient_divider(url, pantry))
    return recipes

def matching_ingredient_df(df):
    """This function takes a dataframe and updates the recipes dataframe with a column that shows the percentage of ingredients in the recipe that are in the urls ingredients list.

    Args:
        df (_type_): _description_
    """
    sample_ingredients = df.ingredients.tolist()
    pantry['in_recipe'] = False
    pantry.loc[pantry.ingredients.isin(sample_ingredients), 'in_recipe'] = True
    recipes['matches'] = recipes.urls.apply(ingredient_divider)

def check_fridge_availability(ingredient, amount_needed, fridge):
    """
    Check if the ingredient is available in the fridge and if there is enough of it.

    Args:
    ingredient (str): The name of the ingredient.
    amount_needed (float): The amount of the ingredient needed.

    Returns:
    float: The amount that still needs to be purchased.
    """
    if ingredient in fridge['ingredients'].values:
        available_amount = fridge[fridge['ingredients'] == ingredient]['amounts'].values[0]
        if available_amount >= amount_needed and amount_needed > 0:
            # Enough available, no need to purchase more
            return 0
        else:
            # Some amount still needs to be purchased
            return amount_needed - available_amount
    else:
        # Ingredient not in fridge, need to purchase the entire amount
        return amount_needed
    
def calculate_cost_and_remaining(row, conversion):
    """
    This function takes a row from a DataFrame and returns the cost of the ingredient and the amount remaining after purchase,
    accounting for unit conversions if necessary.

    Args:
        row (pd.Series): A row from the DataFrame.
        conversion_df (pd.DataFrame): DataFrame containing unit conversion rates.

    Returns:
        pd.Series: A Series containing the calculated cost and remaining amount.
    """
    amount_to_purchase = check_fridge_availability(row['ingredients'], row['amounts'])
    
    # Handling unit conversion
    try:
        if row['units'] != row['purchase_unit']:
            conversion_rate = conversion.loc[(conversion['units'] == row['units']) & (conversion['purchase_unit'] == row['purchase_unit']), 'rate'].iloc[0]
            if conversion_rate != None:
                amount_to_purchase *= conversion_rate
            else:
                pass
    except:
        print(row['ingredients'])
    if row['purchase_size'] == 0 or pd.isna(row['purchase_size']):
        return pd.Series([0, 0], index=['ingredient_cost', 'remaining'])
    
    if amount_to_purchase > 0:
        if amount_to_purchase > row['purchase_size']:
            num_units = np.ceil(amount_to_purchase / row['purchase_size'])
            cost = num_units * row['local_price']
            remaining = (num_units * row['purchase_size']) - amount_to_purchase
        else:
            cost = row['local_price']
            remaining = row['purchase_size'] - amount_to_purchase
    else:
        cost = 0
        remaining = 0
    
    return pd.Series([cost, remaining], index=['ingredient_cost', 'remaining'])

def calculate_cost_of_urls(urls: list, pantry: pd.DataFrame):
    # Assuming combined_df is your DataFrame containing the grocery list
    combined_df = pantry[pantry.urls.isin(urls)].copy()
    # Group by ingredients to sum the amounts needed for each
    grouped_df = combined_df.groupby('ingredients').agg({'amounts': 'sum', 'purchase_size': 'first', 'local_price': 'first', 'units':'first', 'purchase_unit': 'first'}).reset_index()

    # Apply the function to calculate the cost and remaining amount for each ingredient
    grouped_df[['ingredient_cost', 'remaining']] = grouped_df.apply(calculate_cost_and_remaining, axis=1)

    # Summing up the total cost
    total_cost = grouped_df['ingredient_cost'].sum()

    # print(f"Total Cost: {total_cost}")
    grouped_df['cumsum'] = grouped_df['ingredient_cost'].cumsum()
    return grouped_df

def find_recipes(recipes, local_urls, price_cap = 450, dishes = 2, ):
    test_var = True

    tries = 0

    test_df = calculate_cost_of_urls(local_urls)
    current_price = test_df['cumsum'].max()

    while test_var:
        next_url = recipes[(recipes.vegeterian == True) &(recipes.dinner == True) & (~recipes.urls.isin(local_urls))].sort_values(['matches', 'price_add'], ascending=False).head(40).sample(1).urls.values[0]

        local_urls.append(next_url)

        potential_df = calculate_cost_of_urls(local_urls)

        if potential_df['cumsum'].max() < price_cap:
            if len(local_urls) > dishes - 1:
                test_var = False
                # print(local_urls)
        else:
            local_urls.remove(next_url)
            test_var = True

        if tries > 1000:
            print('i tried a thousand times')
            test_var = False

        tries += 1
    return local_urls
 
def set_fridge(potential_df, fridge):
    for ingredient in potential_df.ingredients:
        if isinstance(ingredient, str):
        # if the value is in the fridge we need to subtract the amounts from the fridge
            if ingredient in fridge.ingredients.values:
                amount = potential_df[potential_df.ingredients == ingredient].amounts.values[0]
                if fridge[fridge.ingredients == ingredient].amounts.values[0] == 0:
                    pass
                else:
                    if fridge[fridge.ingredients == ingredient].amounts.values[0] - amount < 0:
                        pass
                    else:
                        fridge.loc[fridge.ingredients == ingredient, 'amounts'] = fridge[fridge.ingredients == ingredient].amounts.values[0] - amount
            if ingredient not in fridge.ingredients.values:
            # if the value is not in the fridge we need to add it to the fridge
                remainder = potential_df[potential_df.ingredients == ingredient].remaining.values[0]
                if remainder == 0:
                    pass
                else:
                    fridge.loc[len(fridge)] = [ingredient, remainder]
    return fridge
