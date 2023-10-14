from cv_agent import CVAgent

if(__name__ == "__main__"):
    ## If DEBUG == None - no debugging
    ## If DEBUG == "console" - show console messages
    ## If DEBUG == "detect" - show detection screen and console messages
    agent = CVAgent(debug=None)

    # Number of steps taken before another action is chosen
    agent.STEPS_PER_ACTION = 4

    # Range (in pixels) between Mario and a Goomba before Mario will jump
    agent.GOOMBA_RANGE = 40

    # Range (in pixels) between Mario and a Koopa before Mario will jump
    agent.KOOPA_RANGE = 50
    
    # If metrics = True, a dictionary containing results of the run are returned
    agent.play(metrics=False)