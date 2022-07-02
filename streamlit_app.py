import streamlit as st

import asyncio
import sys
sys.path.append('src')

from game import Models, Driver, random_deal_source, human, conf

st.text('start')
async def main():
    models = Models.from_conf(conf.load('default.conf'))

    driver = Driver(models, human.ConsoleFactory())

    deal_source = random_deal_source()

    deal_str, auction_str = next(deal_source)
    driver.set_deal(deal_str, auction_str)

    driver.human = [False, False, False, False]
    await driver.run()

asyncio.run(main())
st.text('end')



