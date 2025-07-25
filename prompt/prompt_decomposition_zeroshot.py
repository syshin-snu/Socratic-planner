def DecomQAPrompt(instruction):
    prompt = f'''Suppose you are an embodied agent in the simulation household environment.
        You should decompose the given instruction by questioning and answering yourself.
        
        Rule:      
        1. You can only use Knife when slicing an object. So pickup Knife before slicing, and after you slice the object, you should put knife somewhere.
        2. You can only use a Microwave when heating an object.
        3. Put it in the fridge for a while if you wnat to cool the object.
        4. You can only use a Faucet to clean an object.
        5. All objects are not sliced, not heated, not cooled, not cleaned at the beginning time, if you want, you have to make it like that through a series of actions.
        6. You can not PickUp Two Objects at consecutively and you can't do any other action except Put when you're holding something.
        7. You can put object inside object.
        8. You don't have to use the word in instruction as it is, you should change it according to the set rules and format.
        
        At first, you should split the instruction into sub-tasks.
        Then, you have to decide the order of the sub-tasks.
        After that, for each sub task, you need to determine:
        (1) the object to be performed the sub task.
        (2) the tool objects used to perform the sub task.
        (3) the place object to perform the sub task.
        
        Instruction: an instruction including few sub-tasks that you need to decompose
    
        Starting below, you should follow this format:
        
        Question: asks what sub-tasks you can decompose instruction into.
        Answer: choose the sub-tasks among [Slice, Clean, Heat, Cool, Examine in light, Stack, Pick&Place]
        Question: a question asking the order of sub-tasks.
        Answer: ordering the sub-tasks.
        Question: ask the object to be performed the first sub task 'first sub task'
        Answer: classify the answer into [Books, Ceiling, Door, Floor, KitchenIsland, LightFixture, Rug, Wall, StandardWallSize, Faucet, Bottle, Bag, Cube, Room, AlarmClock, Apple, ArmChair, BaseballBat, BasketBall, Bathtub, BathtubBasin, Bed, Blinds, Book, Boots, Bowl, Box, Bread, ButterKnife, Cabinet, Candle, Cart, CD, CellPhone, Chair, Cloth, CoffeeMachine, CounterTop, CreditCard, Cup, Curtains, Desk, DeskLamp, DishSponge, Drawer, Dresser, Egg, FloorLamp, Footstool, Fork, Fridge, GarbageCan, Glassbottle, HandTowel, HandTowelHolder, HousePlant, Kettle, KeyChain, Knife, Ladle, Laptop, LaundryHamper, LaundryHamperLid, Lettuce, LightSwitch, Microwave, Mirror, Mug, Newspaper, Ottoman, Painting, Pan, PaperTowel, PaperTowelRoll, Pen, Pencil, PepperShaker, Pillow, Plate, Plunger, Poster, Pot, Potato, RemoteControl, Safe, SaltShaker, ScrubBrush, Shelf, ShowerDoor, ShowerGlass, Sink, SinkBasin, SoapBar, SoapBottle, Sofa, Spatula, Spoon, SprayBottle, Statue, StoveBurner, StoveKnob, DiningTable, CoffeeTable, SideTable, TeddyBear, Television, TennisRacket, TissueBox, Toaster, Toilet, ToiletPaper, ToiletPaperHanger, ToiletPaperRoll, Tomato, Towel, TowelHolder, TVStand, Vase, Watch, WateringCan, Window, WineBottle, empty]
        ... (this Question/Answer can repeat N times until the last sub-task)
        
        Begin!
        
        Instruction: {instruction}
        '''
    return prompt



def PolicyPrompt(instruction,QA):
    policy_prompt = f'''Suppose you are an expert embodied agent in the simulation household environment.
            
        Create a low-level plan for completing a household task using the Allowed actions and Allowed objects. 
        You can refer to the questioning and answering conversation which is for decomposing the given instruction to the sub tasks.
        [{QA}]
        In the planning, you should follow the template: (action, object), only if the action is PutObject: (action, object, place)

        Rule:
        1. You can only use Knife when slicing an object. So pickup Knife before slicing, and after you slice the object, you should put knife somewhere.
        2. You can only use a Microwave when heating an object.
        3. Put it in the fridge for a while if you wnat to cool the object.
        4. You can only use a Faucet to clean an object.
        5. All objects are not sliced, not heated, not cooled, not cleaned at the beginning time, if you want, you have to make it like that through a series of actions.
        6. You can not PickUp Two Objects at consecutively and you can't do any other action except Put when you're holding something.
        7. You can put object inside object.
        8. You should CloseObject after OpenObject and you should PickupObject before PutObject, and you should ToggleObjectOff after ToggleObjectOn.
        9. There are also few steps to be taken between sub-tasks.
        
        You don't need any explanation, only create low-level plan to complete the instruction with appropriate template using only Allowed actions and Allowed objects.
        You need to plan according to the order for as detailed steps as possible without missing sub-tasks.
        
        Allowed action: [PickupObject, PutObject, OpenObject, CloseObject, ToggleObjectOn, ToggleObjectOff, SliceObject]
        Allowed object: [Books, Ceiling, Door, Floor, KitchenIsland, LightFixture, Rug, Wall, StandardWallSize, Faucet, Bottle, Bag, Cube, Room, AlarmClock, Apple, ArmChair, BaseballBat, BasketBall, Bathtub, BathtubBasin, Bed, Blinds, Book, Boots, Bowl, Box, Bread, ButterKnife, Cabinet, Candle, Cart, CD, CellPhone, Chair, Cloth, CoffeeMachine, CounterTop, CreditCard, Cup, Curtains, Desk, DeskLamp, DishSponge, Drawer, Dresser, Egg, FloorLamp, Footstool, Fork, Fridge, GarbageCan, Glassbottle, HandTowel, HandTowelHolder, HousePlant, Kettle, KeyChain, Knife, Ladle, Laptop, LaundryHamper, LaundryHamperLid, Lettuce, LightSwitch, Microwave, Mirror, Mug, Newspaper, Ottoman, Painting, Pan, PaperTowel, PaperTowelRoll, Pen, Pencil, PepperShaker, Pillow, Plate, Plunger, Poster, Pot, Potato, RemoteControl, Safe, SaltShaker, ScrubBrush, Shelf, ShowerDoor, ShowerGlass, Sink, SinkBasin, SoapBar, SoapBottle, Sofa, Spatula, Spoon, SprayBottle, Statue, StoveBurner, StoveKnob, DiningTable, CoffeeTable, SideTable, TeddyBear, Television, TennisRacket, TissueBox, Toaster, Toilet, ToiletPaper, ToiletPaperHanger, ToiletPaperRoll, Tomato, Towel, TowelHolder, TVStand, Vase, Watch, WateringCan, Window, WineBottle]
        
        Starting below, you should follow this format:
        1. (action, object)
        2. (action, object)
        ... (this repeat N times until the last plan)
        
        Begin!
        
        Instruction: {instruction}
        Low-level plan:'''
    return policy_prompt

def PolicyPrompt_noQA(instruction):
    policy_prompt = f'''Suppose you are an expert embodied agent in the simulation household environment.
            
        Create a low-level plan for completing a household task using the Allowed actions and Allowed objects. 
        In the planning, you should follow the template: (action, object), only if the action is PutObject: (action, object, place)

        Rule:
        1. You can only use Knife when slicing an object. So pickup Knife before slicing, and after you slice the object, you should put knife somewhere.
        2. You can only use a Microwave when heating an object.
        3. Put it in the fridge for a while if you wnat to cool the object.
        4. You can only use a Faucet to clean an object.
        5. All objects are not sliced, not heated, not cooled, not cleaned at the beginning time, if you want, you have to make it like that through a series of actions.
        6. You can not PickUp Two Objects at consecutively and you can't do any other action except Put when you're holding something.
        7. You can put object inside object.
        8. You can't use the object if it's not in Allowed object, so choose one in Allowd object as similar as possible.
        9. You should CloseObject after OpenObject and you should PickupObject before PutObject, and you should ToggleObjectOff after ToggleObjectOn.
        10. There are also few steps to be taken between sub-tasks.
        
        You don't need any explanation, only create low-level plan to complete the instruction with appropriate template using only Allowed actions and Allowed objects.
        You need to plan according to the order for as detailed steps as possible without missing sub-tasks.
        
        Allowed action: [PickupObject, PutObject, OpenObject, CloseObject, ToggleObjectOn, ToggleObjectOff, SliceObject]
        Allowed object: [Books, Ceiling, Door, Floor, KitchenIsland, LightFixture, Rug, Wall, StandardWallSize, Faucet, Bottle, Bag, Cube, Room, AlarmClock, Apple, ArmChair, BaseballBat, BasketBall, Bathtub, BathtubBasin, Bed, Blinds, Book, Boots, Bowl, Box, Bread, ButterKnife, Cabinet, Candle, Cart, CD, CellPhone, Chair, Cloth, CoffeeMachine, CounterTop, CreditCard, Cup, Curtains, Desk, DeskLamp, DishSponge, Drawer, Dresser, Egg, FloorLamp, Footstool, Fork, Fridge, GarbageCan, Glassbottle, HandTowel, HandTowelHolder, HousePlant, Kettle, KeyChain, Knife, Ladle, Laptop, LaundryHamper, LaundryHamperLid, Lettuce, LightSwitch, Microwave, Mirror, Mug, Newspaper, Ottoman, Painting, Pan, PaperTowel, PaperTowelRoll, Pen, Pencil, PepperShaker, Pillow, Plate, Plunger, Poster, Pot, Potato, RemoteControl, Safe, SaltShaker, ScrubBrush, Shelf, ShowerDoor, ShowerGlass, Sink, SinkBasin, SoapBar, SoapBottle, Sofa, Spatula, Spoon, SprayBottle, Statue, StoveBurner, StoveKnob, DiningTable, CoffeeTable, SideTable, TeddyBear, Television, TennisRacket, TissueBox, Toaster, Toilet, ToiletPaper, ToiletPaperHanger, ToiletPaperRoll, Tomato, Towel, TowelHolder, TVStand, Vase, Watch, WateringCan, Window, WineBottle]
        
        Starting below, you should follow this format:
        1. (action, object)
        2. (action, object)
        ... (this repeat N times until the last plan)
        
        Begin!
        
        Instruction: {instruction}
        Low-level plan:'''
    return policy_prompt
