# Project name

LangChain_checkpoint_Oracle_NoSqlDB_integration

## Getting Started

To save and retrieve LangGraph checkpoints from a LangChain based application in an Oracle NoSQL DB use this library.

### Requirements ###

The following packages are rquired:
 - Python 3.12+
 - Oracle NoSQL Python SDK (Borneo) 5.5.0+
 - LangChain 1.3.1+
 - LangGraph 1.2.1+
 - LangGraph Checkpoint 4.1.0+
 - For Oracle NoSQL Cloud Service you will need OCI 2.175.0+

## Documentation

The LangChan agent has to be created with an ```OracleNoSqlDbCheckpointer``` as in the example below.

The ```OracleNoSqlDbCheckpointer``` has 3 parameters:
  - dbEndpoint - the endpoint of the NoSQL Database, by default uses local on-premises: "http://localhost:8080"
  - tableName - The parent table name to use for checkpoints. It uses 2 tables: a checkpoint parent table by default "checkpoints" and a child table "checkpoint.writes". 
  - drop_tables - Drops the current tables if True. By default is False.

## Examples

The following example shows how to use it in an application:

```python
llm = ChatOpenAI(
        model=os.getenv("EXAMPLE_MODEL", "model name"),
        base_url=os.getenv("EXAMPLE_BASE_URL", "model base URL"),
        #api_key=os.getenv("EXAMPLE_API_KEY", "llm provider key"),
    )
agent = create_agent(
        model=llm,
        system_prompt=(
            "You are a helpful assistant."
        ),
        checkpointer = OracleNoSqlDbCheckpointer(ORACLE_NOSQL_DB_ENDPOINT,
            tableName: str = "checkpoints",
            drop_tables=True),
    )
```

For a full example see [example.py](./example.py).

## Help

- Open an issue in the [Issues page](./issues).
- [Oracle NoSQL Developer Forum](https://community.oracle.com/community/groundbreakers/database/nosql_database).

## Contributing

This project welcomes contributions from the community. Before submitting a pull request, please [review our contribution guide](./CONTRIBUTING.md)

## Security

Please consult the [security guide](./SECURITY.md) for our responsible security vulnerability disclosure process

## License

Copyright (c) 2026 Oracle and/or its affiliates.

