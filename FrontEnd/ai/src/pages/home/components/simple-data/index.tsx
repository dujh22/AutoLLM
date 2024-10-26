import React from 'react';
import { Timeline, App } from 'antd';
import { SaveOutlined } from '@ant-design/icons';
import { ISimpleData } from '@/pages/home/types.ts';
import dayjs from 'dayjs';
import styles from './styles.module.scss';

interface IDataProps {
  dataList: ISimpleData[];
}

const SimpleData: React.FC<IDataProps> = ({ dataList }) => {
  const { message } = App.useApp();

  const onSave = () => {
    message.info('复制到知识库，To be done...');
  };

  const items = dataList.map(item => {
    return {
      label: dayjs(item.dateTime ? new Date(item.dateTime) : new Date()).format('YYYY-MM-DD'),
      children: (
        <div className={styles.item}>
          <div dangerouslySetInnerHTML={{ __html: item.content }}></div>
          <SaveOutlined style={{ marginTop: 6 }} onClick={() => onSave()} />
        </div>
      ),
    };
  });

  console.log(items);

  return <Timeline mode="left" items={items} className={styles.timeline} />;
};

export default SimpleData;
