import os
import re

import dotenv
import psycopg
from hstest import StageTest, CheckResult, dynamic_test
from hstest import TestedProgram

dotenv.load_dotenv()
class RAGTest(StageTest):
    test_data = [
        ("I need details about my order 78a9. Also, what is the return policy for damaged items?", r"return|category|item|damaged|return|refund|30|days"),
    ]

    # first check postgresql connection string
    @dynamic_test
    def test1_PostgresConnection(self):
        connection_string = os.getenv("PGVECTOR_CONNECTION_STRING")
        if not connection_string:
            return CheckResult.wrong("PGVECTOR_CONNECTION_STRING is not set. Please set it to your PostgreSQL connection string.")
        # attempt to extract the database name, user, host, and password from the connection string formatted at PGVECTOR_CONNECTION_STRING="postgresql+psycopg://hyper:hyper2025@localhost:5432/hyperdb"
        match = re.match(r"postgresql\+psycopg://(\w+):(\w+)@([\w.]+):(\d+)/(\w+)", connection_string)

        if not match:
            return CheckResult.wrong("PGVECTOR_CONNECTION_STRING is not in the correct format. Please set it to a valid PostgreSQL connection string.")
        user, password, host, port, dbname = match.groups()
        # create a connection string for psycopg
        conn_string = f"dbname={dbname} user={user} password={password} host={host} port={port}"
        # attempt to connect to the database
        try:
            conn = psycopg.connect(conn_string)
            conn.close()
        except psycopg.OperationalError as e:
            return CheckResult.wrong(f"Could not connect to the database. Encountered: {e}")
        except psycopg.DatabaseError as e:
            return CheckResult.wrong(f"Could not connect to the database. Encountered: {e}")

        except Exception as e:
            return CheckResult.wrong(f"Could not connect to the database. Encountered: {e}")
        return CheckResult.correct()

    @dynamic_test(time_limit=0)
    def test2_RunCode(self):
        for question, expected_output in self.test_data:
            program = TestedProgram("main.py")
            program.start()
            output = program.execute(question)

            if not re.findall(r"policy:|question:|answer:", output, re.IGNORECASE):
                return CheckResult.wrong(f"The output does not match the expected output. Please check your code.")
            if not re.findall(expected_output, output, re.IGNORECASE):
                return CheckResult.wrong(f"The output does not match the expected output. Please check your code.")

            order_details = re.search(r"\[\((.*?)\)\]", output)
            if not order_details:
                return CheckResult.wrong(f"The output does not contain the order details. Are you retrieving the order details from the database?")
            order_details = order_details.group(1).split(", ")
            if not re.search(r"78a9", order_details[1], re.IGNORECASE):
                return CheckResult.wrong(f"The output does not contain the correct order ID. Found: {order_details[1]}")
            if not re.search(r"7b87", order_details[2], re.IGNORECASE):
                return CheckResult.wrong(f"The output does not contain the correct item ID. Found: {order_details[2]}")
            if not re.search(r"USB-C Charging Cable", order_details[3], re.IGNORECASE):
                return CheckResult.wrong(f"The output does not contain the correct item name. Found: {order_details[3]}")
            if not re.search(r"Sports", order_details[4], re.IGNORECASE):
                return CheckResult.wrong(f"The output does not contain the correct item category. Found: {order_details[4]}")

        return CheckResult.correct()


if __name__ == '__main__':
    RAGTest().run_tests()
