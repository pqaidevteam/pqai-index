# PQAI Index

Service for searching vector indexes.

Indexes are searched by passing in a query vector. A list of most similar vectors present in the index are then found and their labels and similarities (or distances) are returned in the response.

## Routes

| Method | Endpoint  | Comments                                      |
| ------ | --------- | --------------------------------------------- |
| `GET`  | `/search` | Search for similar items using a query vector |

## License

The project is open-source under the MIT license.

## Contribute

We welcome contributions.

To make a contribution, please follow these steps:

1. Fork this repository.
2. Create a new branch with a descriptive name
3. Make copy of env file as .env and docker-compose.dev.yml as docker-compose.yml
4. Bring indexer to life `docker-compose up`
5. Make the changes you want and add new tests, if needed
6. Make sure all tests are passing `docker exec -i dev_pqai_indexer_api python -m unittest discover ./tests/`
7. Commit your changes
8. Submit a pull request

## Support

Please create an issue if you need help.