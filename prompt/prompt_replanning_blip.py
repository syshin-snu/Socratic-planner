def ReplanningPrompt(inst, completed, failed, visible, res, cause):
    policy_prompt = f'''Imagine you are an expert agent in a simulated household environment.
        You need to modify the Initial plan by questioning and answering yourself.
        
        ---
        **Sub-task:**
        1. Slice: Grab the knife, cut through the object, and carefully set the knife down after slicing.
        2. Clean: Take the object, place it in the sink, turn on the faucet, and then turn off the faucet to clean it.
        3. Heat: Lift the item, open the microwave, insert it, close the microwave, start it, stop it, and then open the microwave to retrieve the heated item.
        4. Cool: Take the item, open the fridge, place it inside, close the fridge, and then open the fridge to retrieve it.
        5. Examine in light: Hold the object, turn on a lamp or light, and inspect it while carrying the item.
        6. Stack: Pick up the object and place it inside another object to create a stack.
        
        **Rule:**
        Follow the template: (action, object).
        **action space**: [PickupObject, PutObject, SliceObject, OpenObject, ToggleObjectOn, ToggleObjectOff, CloseObject]
        **visible object space**: {visible}
        At the beginning, no objects are sliced, heated, cooled, or cleaned. Achieve these states through a series of actions.
        Remember not to pick up items one after another.
        If the location should be open, such as cabinet or darwer, you should open it first before you putting something in.
        Create a plan using only **action space** and **visible object space**.
        ---
        **Things to discover through question answering**
        1. whether there is a need to modify the plan.
        2. the cause of failure.
        3. how to modify the plan.
        ---
        Instruction: {inst}
        Initial plan:
        {completed}
        
        Feedback: There was a problem from the {failed}. The view the agent is looking at is {res}. The agent is not able to {failed} because {cause}.
        ---
        Starting below, you should follow this format:
        **Question**: {{Ask the things to discover through question answering.}}
        **Answer**: {{Answer the question}}
        ... (Repeat the Question/Answer format N times)
        revised plan: {{revise the plan by reflecting the solution you answered. You don't need any explanation}}
        '''
        
    
    return policy_prompt