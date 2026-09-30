from collections.abc import AsyncIterator, Iterator, Sequence
from logging import config
from typing import Any
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from borneo import DeleteRequest, GetTableRequest, NoSQLHandle, NoSQLHandleConfig, QueryRequest, \
    TableRequest, PutRequest, GetRequest, WriteMultipleRequest, \
    GetResult
from borneo.kv import StoreAccessTokenProvider
import base64


class OracleNoSqlDbCheckpointer(BaseCheckpointSaver):
    # initialize the checkpointer with a database connection and a serializer/deserializer
    def __init__(self, dbEndpoint: str = "http://localhost:8080", 
                 tableName: str = "checkpoints",
                 drop_tables: bool = False):
        super().__init__()
        self._table_name = tableName
        self._table_created = False

        config = NoSQLHandleConfig(dbEndpoint, StoreAccessTokenProvider())
        self._handle = NoSQLHandle(config)
        self.create_table(tableName, drop_tables=drop_tables)

    def create_table(self, table_name: str, drop_tables: bool = False) -> bool:
        """Create the Oracle NoSQL Database table if it doesn't exist.
           Only first time at init, drop the table if drop_tables is True.
        """
        if self._table_created:
            return False  # Table already created, no need to create again
        if not self._handle:
            raise ValueError("NoSQL DB Handle is not initialized.")

        if drop_tables:
            try:
                print(f"   ...Dropping table '{table_name}.writes' if it exists...")
                ddl = f"""DROP TABLE IF EXISTS {table_name}.writes"""
                result = self._handle.table_request(
                    TableRequest()
                    .set_statement(ddl)
                )
                result.wait_for_completion(self._handle, 40000, 3000)
                print(f"Table '{table_name}.writes' dropped.")
                self._table_created = False
            except Exception as e:
                print(f"   ...Error dropping table '{table_name}.writes': {e}")
                self._table_created = False
        
            try:
                print(f"   ...Dropping table '{table_name}' if it exists...")
                ddl = f"""DROP TABLE IF EXISTS {table_name}"""
                result = self._handle.table_request(
                    TableRequest()
                    .set_statement(ddl)
                )
                result.wait_for_completion(self._handle, 40000, 3000)
                print(f"Table '{table_name}' dropped.")
                self._table_created = False
            except Exception as e:
                print(f"   ...Error dropping table '{table_name}': {e}")
                self._table_created = False

        parent_table_created = False
        child_table_created = False
        
        try:
            request = GetTableRequest().set_table_name(table_name)
            self._handle.get_table(request)
            print(f"   ...Table '{table_name}' already exists.")
            parent_table_created = True
        except Exception:
            try:
                print(f"   ...Creating table '{table_name}'...")
                ddl = f"""CREATE TABLE IF NOT EXISTS {table_name} (
                            thread_id     string,
                            checkpoint_ns string,
                            checkpoint_id string,
                            val_        JSON,
                            PRIMARY KEY(SHARD(thread_id), checkpoint_ns, checkpoint_id)
                    )"""
                result = self._handle.table_request(
                    TableRequest()
                    .set_statement(ddl)
                )
                result.wait_for_completion(self._handle, 40000, 3000)
                print(f"Table '{table_name}' created.")
                parent_table_created = True
            except Exception as e:
                print(f"   ...Error creating table '{table_name}': {e}")
                parent_table_created = False

        try:
            request = GetTableRequest().set_table_name(table_name + ".writes")
            self._handle.get_table(request)
            print(f"   ...Table '{table_name}.writes' already exists.")
            child_table_created = True
        except Exception:
            try:
                print(f"   ...Creating table '{table_name}.writes'...")
                ddl = f"""CREATE TABLE IF NOT EXISTS {table_name}.writes (
                            task_id       string,
                            task_path     string,
                            idx           INTEGER,
                            val_        JSON,
                            PRIMARY KEY(task_id, task_path, idx)
                    )"""
                result = self._handle.table_request(
                    TableRequest()
                    .set_statement(ddl)
                )
                result.wait_for_completion(self._handle, 40000, 3000)
                print(f"Table '{table_name}.writes' created.")
                child_table_created = True
            except Exception as e:
                print(f"   ...Error creating table '{table_name}.writes': {e}")
                child_table_created = False

        self._table_created = parent_table_created and child_table_created
        return self._table_created


    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store a checkpoint with its configuration and metadata.

        Args:
            config: Configuration for the checkpoint.
            checkpoint: The checkpoint to store.
            metadata: Additional metadata for the checkpoint.
            new_versions: New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.

        Raises:
            NotImplementedError: Implement this method in your custom checkpoint saver.
        """
        thread_id = config.get("configurable", {}).get("thread_id")
        checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_id = config.get("configurable", {}).get("checkpoint_id")

        type_, blob = self.serde.dumps_typed(checkpoint)
        meta_type, serialized_metadata = self.serde.dumps_typed(metadata)

        print(f"DBG: put: thread_id={thread_id}, checkpoint_ns={checkpoint_ns}, checkpoint_id={checkpoint_id}, parent_id={parent_id}, type_={type_}")
        print(f"DBG: put: blob size={len(blob)}, serialized_metadata size={len(serialized_metadata)}")

        row = {
            "thread_id" : thread_id,
            "checkpoint_ns" : checkpoint_ns,
            "checkpoint_id" : checkpoint_id,
            "val_": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
                "parent_id": parent_id,
                "type": type_,
                "blob": base64.b64encode(blob).decode('utf-8'),
                "metadata_type": meta_type,
                "metadata": base64.b64encode(serialized_metadata).decode('utf-8')
            }
        }

        put_request = PutRequest().set_table_name(self._table_name).set_value(row)

        put_result = self._handle.put(put_request)
        print(f"DBG: put Checkpoint '{checkpoint_id}' written to table '{self._table_name}' with result: {put_result}")

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)


    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config.get("configurable", {}).get("thread_id")
        checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id")

        print(f"DBG: put_writes: thread_id={thread_id}, checkpoint_ns={checkpoint_ns}, checkpoint_id={checkpoint_id}")

        wmReq: WriteMultipleRequest = WriteMultipleRequest()
        wmReq.set_table_name(self._table_name + ".writes")

        for idx, (channel, value) in enumerate(writes):
            type_, blob = self.serde.dumps_typed(value)
            final_idx = WRITES_IDX_MAP.get(channel, idx)

            row = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
                "task_id": task_id,
                "task_path": task_path,
                "idx": final_idx,
                "val_": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                    "task_id": task_id,
                    "task_path": task_path,
                    "idx": final_idx,
                    "channel": channel,
                    "type": type_,
                    "value": base64.b64encode(blob).decode('utf-8')
                }
            }

            wmReq.add(PutRequest().set_table_name(self._table_name + ".writes").set_value(row), False)

        wmRes = self._handle.write_multiple(wmReq)
        print(f"DBG: put_writes result: {wmRes}")


    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self.put_writes(config, writes, task_id, task_path)
        

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = config.get("configurable", {}).get("thread_id")
        checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id")

        print(f"DBG: aget_tuple: thread_id={thread_id}, checkpoint_ns={checkpoint_ns}, checkpoint_id={checkpoint_id}")

        getRes: GetResult = self._handle.get(
            GetRequest().set_table_name(self._table_name).set_key({ 
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id
                })
        )

        row = getRes.get_value()
        if row is None:
            return None

        parent_config = None
        if row["val_"]["parent_checkpoint_id"]:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": row["val_"]["parent_checkpoint_id"],
                }
            }

        checkpoint = self.serde.loads_typed((row["val_"]["type"], row["val_"]["blob"]))
        metadata = self.serde.loads_typed((row["val_"]["metadata_type"], row["val_"]["metadata"]))

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint = checkpoint,
            metadata = metadata,
            #parent_config = RunnableConfig(parent_config["configurable"]) if parent_config else None,
            pending_writes = None,
        )

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Fetch a checkpoint tuple using the given configuration.

        Args:
            config: Configuration specifying which checkpoint to retrieve.

        Returns:
            The requested checkpoint tuple, or `None` if not found.

        Raises:
            NotImplementedError: Implement this method in your custom checkpoint saver.
        """
        thread_id = config.get("configurable", {}).get("thread_id")
        checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id")

        print(f"DBG: get_tuple: thread_id={thread_id}, checkpoint_ns={checkpoint_ns}, checkpoint_id={checkpoint_id}")

        # read checkpoint
        if checkpoint_id:
            key =  { "thread_id": thread_id,
                     "checkpoint_ns": checkpoint_ns,
                     "checkpoint_id": checkpoint_id
                   }
            getRes: GetResult = self._handle.get(
                GetRequest().set_table_name(self._table_name).set_key(key)
            )
            row = getRes.get_value()
            if row is None:
                return None
        else:
            qRes = self._handle.query(
                QueryRequest().set_statement(
                    # f"SELECT * FROM {self._table_name} WHERE thread_id = '{thread_id}' ORDER BY checkpoint_ns, checkpoint_id DESC LIMIT 1"
                    f"SELECT * FROM {self._table_name} \
                      WHERE thread_id = '{thread_id}' and checkpoint_ns = '{checkpoint_ns}' \
                      ORDER BY checkpoint_id DESC LIMIT 1"
                )
            )
            row = qRes.get_results()
            if not row:
                return None
            row = row[0]

        # read writes associated with the checkpoint
        qReq = QueryRequest().set_statement(
            f"SELECT w.task_id, w.val_.channel as channel, w.val_.type as type, w.val_.value as value \
                FROM {self._table_name}.writes as w \
                WHERE thread_id='{thread_id}' AND \
                    checkpoint_ns='{checkpoint_ns}' AND \
                    checkpoint_id='{row["checkpoint_id"]}' \
                ORDER BY task_id, idx"
        )

        pending_writes = []
        while True:
            qRes = self._handle.query(qReq)
            qRecs = qRes.get_results()
            if qReq.is_done():
                break
            else: 
                pending_writes.append([
                    (r["task_id"], r["channel"], self.serde.loads_typed((r["type"], r["value"])))
                    for r in qRecs
                ])
        else:
            pending_writes = []

        checkpoint = self.serde.loads_typed((row["val_"]["type"], base64.b64decode(row["val_"]["blob"].encode('utf-8'))))
        metadata = self.serde.loads_typed((row["val_"]["metadata_type"], base64.b64decode(row["val_"]["metadata"].encode('utf-8'))))


        parent_config = None
        if row["val_"].get("parent_checkpoint_id"):
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": row["val_"].get("parent_checkpoint_id"),
                }
            }

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": row["thread_id"],
                    "checkpoint_ns": row["checkpoint_ns"],
                    "checkpoint_id": row["checkpoint_id"]
                }
            },
            checkpoint = checkpoint,         #row["value"]["blob"],
            metadata = metadata,             #row["value"]["metadata"],
            parent_config = parent_config ,   #RunnableConfig(parent_config["configurable"]) if parent_config else None,
            pending_writes = pending_writes, #None,
        )

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        thread_id = config.get("configurable", {}).get("thread_id")
        checkpoint_id = before.get("configurable", {}).get("checkpoint_id", "")

        print(f"DBG: alist: checkpoint_id={checkpoint_id}")

        qRes = self._handle.query(
            QueryRequest().set_statement(
                f"SELECT * FROM {self._table_name} \
                    WHERE \
                        thread_id = '{thread_id}' AND \
                        checkpoint_id < '{checkpoint_id}' \
                    ORDER BY checkpoint_id DESC LIMIT {limit if limit else 10}"
            )
        )

        while True:
            qRes = self._handle.query(qReq)
            qRecs = qRes.get_results()
            if qReq.is_done():
                break
            else: 
                for r in qRecs:
                    # yield  # make this an async generator
                    yield CheckpointTuple(
                        config={
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": r.get("checkpoint_ns"),
                                "checkpoint_id": r.get("checkpoint_id"),
                            }
                        },
                        checkpoint = None,
                        metadata = None,
                        pending_writes = None,
                    )


    async def adelete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints associated with a specific thread ID."""
        print(f"DBG: adelete_thread: thread_id={thread_id}")

        delRes = self._handle.delete(
            DeleteRequest().set_table_name(self._table_name).set_key({ 
                "thread_id": thread_id
            })
        )

