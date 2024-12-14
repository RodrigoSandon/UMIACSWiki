import requests
import json


class RAGFlow:
    def __init__(self, url, api_key, user_id="user"):
        # self.url = url
        # self.api_key = api_key
        # self.user_id = user_id
        self.req = lambda method, endpoint, json=None, stream=None: requests.request(
            method,
            url + "/v1" + endpoint,
            json=json,
            stream=stream,
            headers={"Authorization": "Bearer " + api_key},
        )

        response = self.req("GET", "/api/new_conversation")
        # print(response.request.body)
        response = response.json()
        self.conversation_id = response["data"]["id"]
        # print(response)

    def __call__(self, input, reset=False, stream=False):  # just dont use stream
        if reset:
            response = self.req("GET", "/api/new_conversation").json()
            self.conversation_id = response["data"]["id"]
        # what is quote option
        if not stream:
            response = self.req(
                "POST",
                "/api/completion",
                {
                    "conversation_id": self.conversation_id,
                    "messages": [{"role": "user", "content": input}],
                    "stream": stream,
                },
            ).json()

            self.last_response = response

            return response["data"]["answer"]
        else:
            raise Exception(
                "stream=True means using yield means this function returns a generator means annoying to work with means commented out"
            )
            # with self.req(
            #     "POST",
            #     "/api/completion",
            #     {
            #         "conversation_id": self.conversation_id,
            #         "messages": [{"role": "user", "content": input}],
            #         # "stream": stream, # ??
            #     },
            #     stream,
            # ) as response:
            #     for line in response.iter_lines():
            #         if line:
            #             j = json.loads(line[5:])["data"]
            #             if j is not True:
            #                 yield j["answer"]


# chat = RAGFlow("http://localhost", "ragflow-FlMGI3NDYyODExYjExZWZiZmJjMDI0Mm")

# res = chat("what nexus partitions are there")

# print(res)
