
public class Ex3_2 {
    public static void main(String[] args) {
        int numOfApples = 123;
        int sizeOfBucket = 10;
        int numOfBucket = (int) Math.ceil((numOfApples*1.0)/sizeOfBucket);
        // int를 int로 나눌 경우, return이 int가 나옴
        // 소수점 이하의 자리가 필요할 경우, 대상 중 하나를 실수로 변경해주는 작업이 필요
        // Math.ceil(): 소수점 첫째자리에서 올림
        
        System.out.println(numOfBucket);

    }
}
