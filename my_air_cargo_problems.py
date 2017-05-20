from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import (
    Node, Problem,
)
from aimacode.utils import expr
from lp_utils import (
    FluentState, encode_state, decode_state,
)
from my_planning_graph import PlanningGraph

from functools import lru_cache

from itertools import product

# to prevent typos, some handy functions for generating expressions:
def expr_at(plane_or_cargo, airport) -> expr:
    return expr("At({}, {})".format(plane_or_cargo, airport))


def expr_in(cargo, plane) -> expr:
    return expr("In({}, {})".format(cargo, plane))


def expr_load(cargo, plane, airport) -> expr:
    return expr('Load({}, {}, {})'.format(cargo, plane, airport))


def expr_unload(cargo, plane, airport) -> expr:
    return expr('Unload({}, {}, {})'.format(cargo, plane, airport))


def expr_fly(plane, fr, to) -> expr:
    return expr("Fly({}, {}, {})".format(plane, fr, to))


class AirCargoProblem(Problem):
    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        """

        :param cargos: list of str
            cargos in the problem
        :param planes: list of str
            planes in the problem
        :param airports: list of str
            airports in the problem
        :param initial: FluentState object
            positive and negative literal fluents (as expr) describing initial state
        :param goal: list of expr
            literal fluents required for goal test
        """
        self.state_map = initial.pos + initial.neg
        self.initial_state_TF = encode_state(initial, self.state_map)
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        self.cargos = cargos
        self.planes = planes
        self.airports = airports
        self.actions_list = self.get_actions()

    def get_actions(self):
        """
        This method creates concrete actions (no variables) for all actions in the problem
        domain action schema and turns them into complete Action objects as defined in the
        aimacode.planning module. It is computationally expensive to call this method directly;
        however, it is called in the constructor and the results cached in the `actions_list` property.

        Returns:
        ----------
        list<Action>
            list of Action objects
        """

        # TODO create concrete Action objects based on the domain action schema for: Load, Unload, and Fly
        # concrete actions definition: specific literal action that does not include variables as with the schema
        # for example, the action schema 'Load(c, p, a)' can represent the concrete actions 'Load(C1, P1, SFO)'
        # or 'Load(C2, P2, JFK)'.  The actions for the planning problem must be concrete because the problems in
        # forward search and Planning Graphs must use Propositional Logic

        def load_actions():
            """Create all concrete Load actions and return a list

            :return: list of Action objects
            """
            loads = []
            # TODO create all load ground actions from the domain Load action
            for (airport, plane, cargo) in product(self.airports, self.planes, self.cargos):
                precond_pos = [expr_at(plane, airport),
                               expr_at(cargo, airport)]
                effect_add = [expr_in(cargo, plane)]
                effect_rem = [expr_at(cargo, airport)]
                loads.append(Action(expr_load(cargo, plane, airport),
                        [precond_pos, []],
                        [effect_add, effect_rem]))
            return loads

        def unload_actions():
            """Create all concrete Unload actions and return a list

            :return: list of Action objects
            """
            unloads = []
            # TODO create all Unload ground actions from the domain Unload action
            for (airport, plane, cargo) in product(self.airports, self.planes, self.cargos):
                precond_pos = [expr_at(plane, airport),
                               expr_in(cargo, plane)]
                effect_add = [expr_at(cargo, airport)]
                effect_rem = [expr_in(cargo, plane)]
                unloads.append(Action(expr_unload(cargo, plane, airport),
                        [precond_pos, []],
                        [effect_add, effect_rem]))
            return unloads

        def fly_actions():
            """Create all concrete Fly actions and return a list

            :return: list of Action objects
            """
            flys = []
            for fr in self.airports:
                for to in self.airports:
                    if fr != to:
                        for p in self.planes:
                            precond_pos = [expr_at(p, fr)]
                            precond_neg = []
                            effect_add = [expr_at(p, to)]
                            effect_rem = [expr_at(p, fr)]
                            fly = Action(expr_fly(p, fr, to),
                                         [precond_pos, precond_neg],
                                         [effect_add, effect_rem])
                            flys.append(fly)
            return flys

        all = load_actions() + unload_actions() + fly_actions()
        return all

    def actions(self, state: str) -> list:
        """ Return the actions that can be executed in the given state.

        :param state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'
        :return: list of Action objects
        """
        # TODO implement
        possible_actions = []
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for action in self.actions_list:
            is_possible = True
            for clause in action.precond_pos:
                if clause not in kb.clauses:
                    is_possible = False
            for clause in action.precond_neg:
                if clause in kb.clauses:
                    is_possible = False
            if is_possible:
                possible_actions.append(action)

        return possible_actions

    def result(self, state: str, action: Action):
        """ Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).

        :param state: state entering node
        :param action: Action applied
        :return: resulting state after action
        """
        # TODO implement
        # start with setting the new_state to what directly happens as an effect of an action
        new_state = FluentState([], [])

        already_added = action.effect_add + action.effect_rem
        # transfer data from old state to new state
        old_state = decode_state(state, self.state_map)
        new_state.pos = action.effect_add + [f for f in old_state.pos if fluent not in already_added]
        new_state.neg = action.effect_rem + [f for f in old_state.neg if fluent not in already_added]

        return encode_state(new_state, self.state_map)

    def goal_test(self, state: str) -> bool:
        """ Test the state to see if goal is reached

        :param state: str representing state
        :return: bool
        """
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())

        for clause in self.goal:
            if clause not in kb.clauses:
                return False
        return True

    def h_1(self, node: Node):
        # note that this is not a true heuristic
        h_const = 1
        return h_const

    @lru_cache(maxsize=8192)
    def h_pg_levelsum(self, node: Node):
        """This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        """
        # requires implemented PlanningGraph class
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum

    @lru_cache(maxsize=8192)
    def h_ignore_preconditions(self, node: Node):
        """This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.
        """
        # TODO implement (see Russell-Norvig Ed-3 10.2.3  or Russell-Norvig Ed-2 11.2)
        # basing on https://discussions.udacity.com/t/understanding-ignore-precondition-heuristic/225906/2
        kb = PropKB()
        kb.tell(decode_state(node.state, self.state_map).pos_sentence())

        count = sum([1 for clause in self.goal if clause not in kb.clauses])
        return count


def air_cargo_p1() -> AirCargoProblem:
    cargos = ['C1', 'C2']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
    neg = [expr('At(C2, SFO)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('At(C1, JFK)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('At(P1, JFK)'),
           expr('At(P2, SFO)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p2() -> AirCargoProblem:
    # TODO implement Problem 2 definition
    cargos = ['C1', 'C2', 'C3']
    planes = ['P1', 'P2', 'P3']
    airports = ['JFK', 'SFO', 'ATL']

    # generate list of all fluents
    all = set(expr_in(c, p) for (c, p) in product(cargos, planes)) \
          | set(expr_at(p, a) for (p, a) in product(planes, airports)) \
          | set(expr_at(c, a) for (c, a) in product(cargos, airports))

    pos = [expr_at('C1', 'SFO'), expr_at('C2', 'JFK'), expr_at('C3', 'ATL'),
           expr_at('P1', 'SFO'), expr_at('P2', 'JFK'), expr_at('P3', 'ATL')]
    init = FluentState(pos, list(all - set(pos)))
    goal = [expr_at('C1', 'JFK'), expr_at('C2', 'SFO'), expr_at('C3', 'SFO')]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p3() -> AirCargoProblem:
    # TODO implement Problem 3 definition
    cargos = ['C1', 'C2', 'C3', 'C4']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO', 'ATL', 'ORD']

    # generate list of all fluents
    all = set(expr_in(c, p) for (c, p) in product(cargos, planes)) \
          | set(expr_at(p, a) for (p, a) in product(planes, airports)) \
          | set(expr_at(c, a) for (c, a) in product(cargos, airports))

    pos = [expr_at('C1', 'SFO'), expr_at('C2', 'JFK'), expr_at('C3', 'ATL'), expr_at('C4', 'ORD'),
           expr_at('P1', 'SFO'), expr_at('P2', 'JFK')]
    init = FluentState(pos, list(all - set(pos)))
    goal = [expr_at('C1', 'JFK'), expr_at('C2', 'SFO'), expr_at('C3', 'JFK'), expr_at('C4', 'SFO')]
    return AirCargoProblem(cargos, planes, airports, init, goal)
