#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Miao at 2025-04-04
'''
@File   : const_db_fields
@Time   : 2025-04-04 3:00 p.m.
@Author : Miao
@Project: IFT6289-multimodal-rag 
@Desc   : The const values searched from mongodb, used for the gradio interface to let the user choose.
'''

PROPERTY_TYPES = ["Aparthotel", "Apartment", "Barn", "Bed and breakfast", "Boat", "Boutique hotel", "Bungalow", "Cabin", "Camper/RV", "Campsite", "Casa particular (Cuba)", "Castle", "Chalet", "Condominium", "Cottage", "Earth house", "Farm stay", "Guest suite", "Guesthouse", "Heritage hotel (India)", "Hostel", "Hotel", "House", "Houseboat", "Hut", "Loft", "Nature lodge", "Other", "Pension (South Korea)", "Resort", "Serviced apartment", "Tiny house", "Townhouse", "Train", "Treehouse", "Villa"]

if __name__=='__main__':
    from utils.mongodb import get_collection

    def get_unique_values(filed_name='property_type'):
        collection = get_collection()
        return collection.distinct(filed_name)