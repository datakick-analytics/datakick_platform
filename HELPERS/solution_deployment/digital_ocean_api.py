class DigitalOcean(object):
    images_url = 'https://api.digitalocean.com/v2/images'
    droplets_url = 'https://api.digitalocean.com/v2/droplets'
    databases_url = 'https://api.digitalocean.com/v2/databases'

    def __init__(self, api_key):
        self.headers = {
            'content-type': 'application/json',
            'authorization': 'Bearer {}'.format(api_key)
        }

    def get_image_list(self):
        req = Request(url='{}'.format(self.images_url),
                      headers=self.headers)

        try:
            with urlopen(req) as response:
                return json.loads(response.read().decode('utf8'))

        except Exception as e:
            print(e)

    def get_database_list(self):
        req = Request(url='{}'.format(self.databases_url),
                      headers=self.headers)

        try:
            with urlopen(req) as response:
                return json.loads(response.read().decode('utf8'))

        except Exception as e:
            print(e)

    def get_database(self, database_id):
        req = Request(url='{}/{}'.format(self.databases_url, database_id),
                      headers=self.headers)

        try:
            with urlopen(req) as response:
                return json.loads(response.read().decode('utf8'))

        except Exception as e:
            print(e)

    def update_db_firewall(self, database_id, entries):
        values = {'rules': entries}
        data = json.dumps(values).encode('utf8')

        req = Request(url='{}/{}/firewall'.format(self.databases_url, database_id),
                      data=data, headers=self.headers, method='PUT')

        try:
            with urlopen(req) as response:
                return response.read().decode('utf8')

        except Exception as e:
            print(e)

    def get_droplet_list(self):
        req = Request(url='{}'.format(self.droplets_url), headers=self.headers)

        try:
            with urlopen(req) as response:
                return json.loads(response.read().decode('utf8'))

        except Exception as e:
            print(e)

    def create_droplet(self, name, region, size, image, ssh_keys, private_networking, backups, user_data):
        values = {'name': name, 'region': region, 'size': size, 'image': image, 'ssh_keys': ssh_keys,
                  'private_networking': private_networking, 'backups': backups, 'user_data': user_data}
        data = json.dumps(values).encode('utf8')

        req = Request(self.droplets_url, data=data, headers=self.headers)

        try:
            with urlopen(req) as response:
                return json.loads(response.read().decode('utf8'))

        except Exception as e:
            print(e)

    def delete_droplet(self, droplet_id):
        req = Request(url='{}/{}'.format(self.droplets_url, droplet_id),
                      headers=self.headers, method='DELETE')

        try:
            with urlopen(req) as response:
                return response.getcode()

        except Exception as e:
            print(e)

    def simplify(self, result):
        return [
            {
                'id': entry.get('id'),
                'name': entry.get('name'),
                'status': entry.get('status')
            }

            for entry in result.get('droplets')
        ]